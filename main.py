from concurrent.futures import ThreadPoolExecutor
import atexit
import os
import google.generativeai as genai
import requests
from flask import Flask, request, jsonify
import pathlib
import json
import tempfile
import logging
import traceback
from json_repair import repair_json  # Import json-repair as shown in docs
from json import JSONDecodeError

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize Gemini API client with your API key
gemini_api_key = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=gemini_api_key)
airtable_webhook_url = os.getenv('AIRTABLE_WEBHOOK', 'http://your-airtable-webhook-url')

# Create a thread pool executor for tasks
executor = ThreadPoolExecutor(max_workers=1)

def download_pdf(pdf_url, download_folder):
    try:
        download_folder = pathlib.Path(download_folder)
        download_folder.mkdir(parents=True, exist_ok=True)
        file_path = download_folder / 'downloaded_pdf.pdf'

        response = requests.get(pdf_url, timeout=60)
        response.raise_for_status()

        with open(file_path, 'wb') as file:
            file.write(response.content)

        logger.info(f"Downloaded PDF to: {file_path}")
        return str(file_path)
    except requests.RequestException as e:
        logger.error(f"Error downloading PDF from {pdf_url}: {str(e)}")
        raise

def upload_pdf_to_gemini(pdf_path):
    try:
        file_ref = genai.upload_file(str(pdf_path))
        logger.info(f"Successfully uploaded {pdf_path} to Gemini")
        return file_ref
    except Exception as e:
        logger.error(f"Error uploading PDF to Gemini: {str(e)}")
        raise

def extract_text_with_gemini(file_ref, text_extraction_prompt):
    try:
        model = genai.GenerativeModel(model_name='gemini-1.5-flash')

        response = model.generate_content([file_ref, text_extraction_prompt])
        logger.info(f"Response from Gemini: {response}")

        if response.candidates and response.candidates[0].content.parts:
            extracted_text = response.candidates[0].content.parts[0].text
            logger.info(f"Extracted text from PDF: {extracted_text}")
            return extracted_text
        else:
            logger.warning("No text extracted from the PDF.")
            return ""
    except Exception as e:
        logger.error(f"Error in extract_text_with_gemini: {str(e)}")
        raise

def validate_and_repair_json(json_content):
    try:
        parsed_json = json.loads(json_content)
        return json.dumps(parsed_json)
    except JSONDecodeError as e:
        logger.warning("JSON is malformed; attempting to repair.")
        try:
            repaired_json = repair_json(json_content)
            parsed_json = json.loads(repaired_json)
            return json.dumps(parsed_json)
        except JSONDecodeError as repair_error:
            logger.error(f"Failed to repair JSON: {repair_error}")
            logger.error(f"Context around error: {json_content[max(0, e.pos - 40):e.pos + 40]}")
            return ""

def generate_marking_guide_with_gemini(file_ref, marking_guide_prompt):
    try:
        model = genai.GenerativeModel(model_name='gemini-1.5-flash')
        response = model.generate_content([file_ref, marking_guide_prompt])
        if response.candidates and response.candidates[0].content.parts:
            marking_guide = response.candidates[0].content.parts[0].text.strip()
            logger.info(f"Generated marking guide: {marking_guide}")
            return marking_guide
        else:
            logger.warning("No marking guide generated.")
            return ""
    except Exception as e:
        logger.error(f"Error generating marking guide: {str(e)}")
        raise

def summarize_content_with_gemini(file_ref, custom_prompt, response_schema, assessment_type_prompt, assessment_name_prompt, marking_guide_prompt):
    try:
        model = genai.GenerativeModel(model_name='gemini-1.5-flash')

        # Generate the marking guide first
        marking_guide = generate_marking_guide_with_gemini(file_ref, marking_guide_prompt)

        if not marking_guide:
            logger.warning("No marking guide generated.")
            return "", "", "", ""

        # Use the generated marking guide as part of the custom prompt for JSON extraction
        json_prompt = f"{custom_prompt}\n\nUse the following marking guide to extract the information according to the schema:\n\n{marking_guide}\n\nSchema:\n{json.dumps(response_schema, indent=2)}"

        # Generate JSON content using the marking guide as input
        json_response = model.generate_content([file_ref, json_prompt])
        type_response = model.generate_content([file_ref, assessment_type_prompt])
        name_response = model.generate_content([file_ref, assessment_name_prompt])

        if json_response.candidates and json_response.candidates[0].content.parts:
            raw_json_content = json_response.candidates[0].content.parts[0].text
            json_content = validate_and_repair_json(raw_json_content)
            logger.info(f"Validated and repaired JSON content: {json_content}")
        else:
            logger.warning("No JSON content extracted.")
            json_content = ""

        # Determine the assessment type
        if type_response.candidates and type_response.candidates[0].content.parts and "Essay" in type_response.candidates[0].content.parts[0].text:
            assessment_type = "Essay"
        elif type_response.candidates and type_response.candidates[0].content.parts and "Exam style" in type_response.candidates[0].content.parts[0].text:
            assessment_type = "Exam style"
        else:
            assessment_type = "Exam style"

        logger.info(f"Determined assessment type: {assessment_type}")

        # Determine the assessment name
        if name_response.candidates and name_response.candidates[0].content.parts:
            assessment_name = name_response.candidates[0].content.parts[0].text.strip()
            logger.info(f"Generated assessment name: {assessment_name}")
        else:
            assessment_name = "Unknown"

        return json_content, assessment_type, assessment_name, marking_guide

    except Exception as e:
        logger.error(f"Error in summarize_content_with_gemini: {str(e)}")
        raise

def send_to_airtable(record_id, json_content, assessment_type, assessment_name, extracted_text, new_marking_guide, target_field_id, status_message):
    try:
        data = {
            "record_id": record_id,
            "json_content": json_content,
            "assessmentType": assessment_type,
            "assessmentName": assessment_name,
            "extracted_text": extracted_text,
            "new_marking_guide": new_marking_guide,
            "status_message": status_message,
            "target_field_id": target_field_id
        }

        response = requests.post(airtable_webhook_url, json=data)
        response.raise_for_status()
        logger.info("Successfully sent data to Airtable")
    except requests.RequestException as e:
        logger.error(f"Error sending data to Airtable: {str(e)}")
        raise

def process_pdf_async_assessment(pdf_url, record_id, custom_prompt, response_schema, text_extraction_prompt, target_field_id, assessment_type_prompt, assessment_name_prompt, marking_guide_prompt):
    def process():
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                request_dir = pathlib.Path(temp_dir)

                pdf_path = download_pdf(pdf_url, request_dir)
                file_ref = upload_pdf_to_gemini(pdf_path)
                extracted_text = extract_text_with_gemini(file_ref, text_extraction_prompt)

                json_content, assessment_type, assessment_name, new_marking_guide = summarize_content_with_gemini(
                    file_ref, custom_prompt, response_schema, assessment_type_prompt, assessment_name_prompt, marking_guide_prompt)
                send_to_airtable(record_id, json_content, assessment_type, assessment_name, extracted_text, new_marking_guide, target_field_id, "Successfully processed by Gemini")

        except Exception as e:
            error_message = f"An error occurred during processing: {str(e)}"
            logger.error(error_message)
            logger.error(traceback.format_exc())
            send_to_airtable(record_id, "", "", "", "", "", target_field_id, error_message)

    executor.submit(process)

def process_pdf_async_submission(pdf_url, record_id, custom_prompt, response_schema, text_extraction_prompt, target_field_id):
    def process():
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                request_dir = pathlib.Path(temp_dir)

                pdf_path = download_pdf(pdf_url, request_dir)
                file_ref = upload_pdf_to_gemini(pdf_path)
                extracted_text = extract_text_with_gemini(file_ref, text_extraction_prompt)

                json_content, _, _, _ = summarize_content_with_gemini(
                    file_ref, custom_prompt, response_schema, "", "", "")
                send_to_airtable(record_id, json_content, "", "", extracted_text, "", target_field_id, "Successfully processed submission by Gemini")

        except Exception as e:
            error_message = f"An error occurred during submission processing: {str(e)}"
            logger.error(error_message)
            logger.error(traceback.format_exc())
            send_to_airtable(record_id, "", "", "", "", "", target_field_id, error_message)

    executor.submit(process)

@app.route('/process_pdf/assessment', methods=['POST'])
def process_pdf_assessment_route():
    try:
        data = request.json
        pdf_url = data.get('pdf_url')
        record_id = data.get('record_id')
        custom_prompt = data.get('custom_prompt')
        response_schema = data.get('response_schema')
        text_extraction_prompt = data.get('text_extraction_prompt')
        target_field_id = data.get('targetFieldId')
        assessment_type_prompt = data.get('assessment_type_prompt')
        assessment_name_prompt = data.get('assessment_name_prompt')
        marking_guide_prompt = data.get('marking_guide_prompt')
        temperature = data.get('temperature', 0)  # Default to 0 if not sent

        if isinstance(response_schema, str):
            response_schema = json.loads(response_schema)

        if pdf_url and record_id and response_schema and text_extraction_prompt and target_field_id:
            process_pdf_async_assessment(pdf_url, record_id, custom_prompt, response_schema, text_extraction_prompt, target_field_id, assessment_type_prompt, assessment_name_prompt, marking_guide_prompt, temperature)
            return jsonify({"status": "processing started"}), 200
        else:
            missing_fields = [field for field in ['pdf_url', 'record_id', 'response_schema', 'text_extraction_prompt', 'targetFieldId', 'assessment_type_prompt', 'assessment_name_prompt', 'marking_guide_prompt'] if not locals().get(field)]
            error_message = f"Missing required fields: {', '.join(missing_fields)}"
            logger.error(error_message)
            return jsonify({"error": error_message}), 400
    except json.JSONDecodeError as e:
        error_message = f"Invalid JSON format in request body or response_schema: {str(e)}"
        logger.error(error_message)
        return jsonify({"error": error_message}), 400
    except Exception as e:
        error_message = f"An unexpected error occurred: {str(e)}"
        logger.error(error_message)
        logger.error(traceback.format_exc())
        return jsonify({"error": error_message}), 500

@app.route('/process_pdf/submission', methods=['POST'])
def process_pdf_submission_route():
    try:
        data = request.json
        pdf_url = data.get('pdf_url')
        record_id = data.get('record_id')
        custom_prompt = data.get('custom_prompt')
        response_schema = data.get('response_schema')
        text_extraction_prompt = data.get('text_extraction_prompt')
        target_field_id = data.get('targetFieldId')
        temperature = data.get('temperature', 0)  # Default to 0 if not sent

        if isinstance(response_schema, str):
            response_schema = json.loads(response_schema)

        if pdf_url and record_id and response_schema and text_extraction_prompt and target_field_id:
            process_pdf_async_submission(pdf_url, record_id, custom_prompt, response_schema, text_extraction_prompt, target_field_id, temperature)
            return jsonify({"status": "submission processing started"}), 200
        else:
            missing_fields = [field for field in ['pdf_url', 'record_id', 'response_schema', 'text_extraction_prompt', 'targetFieldId'] if not locals().get(field)]
            error_message = f"Missing required fields: {', '.join(missing_fields)}"
            logger.error(error_message)
            return jsonify({"error": error_message}), 400
    except json.JSONDecodeError as e:
        error_message = f"Invalid JSON format in request body or response_schema: {str(e)}"
        logger.error(error_message)
        return jsonify({"error": error_message}), 400
    except Exception as e:
        error_message = f"An unexpected error occurred: {str(e)}"
        logger.error(error_message)
        logger.error(traceback.format_exc())
        return jsonify({"error": error_message}), 500

atexit.register(executor.shutdown)

if __name__ == '__main__':
    app.run(debug=True)

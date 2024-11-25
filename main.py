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

def log_payload(data):
    logger.info(f"Received Payload: {json.dumps(data, indent=2)}")

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
        logger.info(f"Successfully uploaded {pdf_path} to Gemini. File reference: {file_ref}")
        return file_ref
    except Exception as e:
        logger.error(f"Error uploading PDF to Gemini: {str(e)}")
        raise

def extract_text_with_gemini(file_ref, text_extraction_prompt, temperature=0):
    try:
        logger.info(f"Extracting text with prompt: {text_extraction_prompt}")
        model = genai.GenerativeModel(model_name='gemini-1.5-flash')
        temperature = float(temperature)
        generation_config = genai.types.GenerationConfig(temperature=temperature)
        response = model.generate_content([file_ref, text_extraction_prompt], generation_config=generation_config)
        logger.info(f"Response from Gemini: {response}")

        if response.candidates and response.candidates[0].content.parts:
            extracted_text = response.candidates[0].content.parts[0].text
            logger.info(f"Extracted text snippet: {extracted_text[:500]}...")  # Log first 500 characters
            return extracted_text
        else:
            logger.warning("No text extracted from the PDF.")
            return ""
    except Exception as e:
        logger.error(f"Error in extract_text_with_gemini: {str(e)}")
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
        logger.info(f"Sending to Airtable: {json.dumps(data, indent=2)}")

        response = requests.post(airtable_webhook_url, json=data)
        response.raise_for_status()
        logger.info("Successfully sent data to Airtable")
    except requests.RequestException as e:
        logger.error(f"Error sending data to Airtable: {str(e)}")
        raise

def process_pdf_async_submission(pdf_url, record_id, custom_prompt, response_schema, text_extraction_prompt, target_field_id, temperature=0):
    def process():
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                logger.info(f"Processing submission for Record ID: {record_id}")

                pdf_path = download_pdf(pdf_url, temp_dir)
                file_ref = upload_pdf_to_gemini(pdf_path)
                extracted_text = extract_text_with_gemini(file_ref, text_extraction_prompt, temperature)

                if not extracted_text:
                    raise ValueError("No text extracted from the PDF. Cannot proceed with JSON generation.")

                updated_custom_prompt = f"{custom_prompt}\n\nExtracted Text:\n{extracted_text}\n\nSchema:\n{json.dumps(response_schema, indent=2)}"
                logger.info(f"Updated custom prompt for JSON extraction: {updated_custom_prompt[:500]}...")  # Log first 500 characters

                json_content = "{}"  # Placeholder if extraction fails
                try:
                    model = genai.GenerativeModel(model_name='gemini-1.5-flash')
                    generation_config = genai.types.GenerationConfig(temperature=float(temperature))
                    json_response = model.generate_content([file_ref, updated_custom_prompt], generation_config=generation_config)

                    if json_response.candidates and json_response.candidates[0].content.parts:
                        raw_json_content = json_response.candidates[0].content.parts[0].text
                        json_content = validate_and_repair_json(raw_json_content)
                        logger.info(f"Validated and repaired JSON content: {json_content}")
                except Exception as e:
                    logger.error(f"Error generating JSON content: {str(e)}")
                    json_content = ""

                send_to_airtable(record_id, json_content, "", "", extracted_text, "", target_field_id, "Successfully processed submission by Gemini")

        except Exception as e:
            error_message = f"An error occurred during submission processing: {str(e)}"
            logger.error(error_message)
            logger.error(traceback.format_exc())
            send_to_airtable(record_id, "", "", "", "", "", target_field_id, error_message)

    executor.submit(process)

@app.route('/process_pdf/submission', methods=['POST'])
def process_pdf_submission_route():
    try:
        data = request.json
        log_payload(data)

        pdf_url = data.get('pdf_url')
        record_id = data.get('record_id')
        custom_prompt = data.get('custom_prompt')
        response_schema = data.get('response_schema')
        text_extraction_prompt = data.get('text_extraction_prompt')
        target_field_id = data.get('targetFieldId')
        temperature = data.get('temperature', 0)

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

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
    """
    Download a PDF file from the given URL and save it to the specified download folder.
    """
    try:
        download_folder = pathlib.Path(download_folder)
        download_folder.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
        file_path = download_folder / 'downloaded_pdf.pdf'

        response = requests.get(pdf_url, timeout=60)  # Set a timeout for the request
        response.raise_for_status()  # Raise an error if the request failed

        with open(file_path, 'wb') as file:
            file.write(response.content)

        logger.info(f"Downloaded PDF to: {file_path}")
        return str(file_path)
    except requests.RequestException as e:
        logger.error(f"Error downloading PDF from {pdf_url}: {str(e)}")
        raise

def upload_pdf_to_gemini(pdf_path):
    """
    Upload a PDF file to the Gemini API.
    """
    try:
        file_ref = genai.upload_file(str(pdf_path))
        logger.info(f"Successfully uploaded {pdf_path} to Gemini")
        return file_ref
    except Exception as e:
        logger.error(f"Error uploading PDF to Gemini: {str(e)}")
        raise

def extract_text_with_gemini(file_ref, text_extraction_prompt):
    """
    Extract text from the PDF using the Gemini API.
    """
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
def summarize_content_with_gemini(file_ref, custom_prompt, response_schema):
    """
    Extract JSON content, determine assessment type, and generate assessment name using the Gemini API.
    """
    try:
        model = genai.GenerativeModel(model_name='gemini-1.5-flash')

        # Separate the custom prompt for JSON extraction, type, and name
        json_prompt = f"{custom_prompt}\n\nPlease extract the information according to the following schema:\n\n{json.dumps(response_schema, indent=2)}"
        assessment_type_prompt = (
            "Your job is to analyse the extracted text and decide which of the following options this assessment meets, in terms of assessment type. "
            "Essay: You should select this option if you think the assessment is an essay-style assessment that probably contains only one question. "
            "Exam style: You should select this option if you think the assessment is an exam-style assessment. "
            "The definition of this is that it contains multiple questions that the student has to answer, and likely has question numbering (e.g., 1.1, 1.2, etc.). "
            "The output that you give me should simply be the assessment type that you think is most likely. "
            "Therefore your output should simply either be: 'Essay' OR 'Exam style'."
        )
        assessment_name_prompt = (
            "Please give this assessment an appropriate name. The title of the assessment might be in the text already, "
            "but if not, provide an intuitive name for the assessment using the content provided. "
            "If you can identify the subject and the grade in the text provided, please include that in the name. "
            "In your output, please only provide the name of the assessment - no pre-text or post-text should exist in your output."
        )

        # Generate content separately for JSON, type, and name
        json_response = model.generate_content([file_ref, json_prompt])
        type_response = model.generate_content([file_ref, assessment_type_prompt])
        name_response = model.generate_content([file_ref, assessment_name_prompt])

        # Extract the JSON content
        if json_response.candidates and json_response.candidates[0].content.parts:
            json_content = json_response.candidates[0].content.parts[0].text
            logger.info(f"Extracted JSON content: {json_content}")
        else:
            logger.warning("No JSON content extracted.")
            json_content = ""

        # Extract the assessment type
        if type_response.candidates and "Essay" in type_response.candidates[0].content.parts[0].text:
            assessment_type = "Essay"
        elif type_response.candidates and "Exam style" in type_response.candidates[0].content.parts[0].text:
            assessment_type = "Exam style"
        else:
            assessment_type = "Unknown"

        logger.info(f"Determined assessment type: {assessment_type}")

        # Extract the assessment name
        if name_response.candidates and name_response.candidates[0].content.parts:
            assessment_name = name_response.candidates[0].content.parts[0].text.strip()
            logger.info(f"Generated assessment name: {assessment_name}")
        else:
            assessment_name = "Unknown"

        return json_content, assessment_type, assessment_name

    except Exception as e:
        logger.error(f"Error in summarize_content_with_gemini: {str(e)}")
        raise

def send_to_airtable(record_id, json_content, assessment_type, assessment_name, target_field_id):
    """
    Send the processed JSON content, assessment type, and assessment name to the Airtable webhook.
    """
    try:
        data = {
            "record_id": record_id,
            "json_content": json_content,  # JSON data
            "assessmentType": assessment_type,  # Assessment type
            "assessmentName": assessment_name,  # Assessment name
            "target_field_id": target_field_id
        }

        response = requests.post(airtable_webhook_url, json=data)
        response.raise_for_status()
        logger.info("Successfully sent data to Airtable")
    except requests.RequestException as e:
        logger.error(f"Error sending data to Airtable: {str(e)}")
        raise

def process_pdf_async_assessment(pdf_url, record_id, custom_prompt, response_schema, text_extraction_prompt, target_field_id):
    def process():
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                request_dir = pathlib.Path(temp_dir)

                # Download the PDF (using the original download_pdf function)
                pdf_path = download_pdf(pdf_url, request_dir)

                # Upload the PDF to Gemini API
                file_ref = upload_pdf_to_gemini(pdf_path)

                # Extract the JSON, assessment type, and assessment name
                json_content, assessment_type, assessment_name = summarize_content_with_gemini(file_ref, custom_prompt, response_schema)

                # Send the JSON, assessment type, and name separately to Airtable
                send_to_airtable(record_id, json_content, assessment_type, assessment_name, target_field_id)

        except Exception as e:
            error_message = f"An error occurred during processing: {str(e)}"
            logger.error(error_message)
            logger.error(traceback.format_exc())
            send_to_airtable(record_id, {"error": error_message}, "", "", target_field_id)

    # Submit the task to the thread pool
    executor.submit(process)

def process_pdf_async_submission(pdf_url, record_id, custom_prompt, response_schema, text_extraction_prompt, target_field_id):
    def process():
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                request_dir = pathlib.Path(temp_dir)

                # Download the PDF
                pdf_path = download_pdf(pdf_url, request_dir)

                # Upload the PDF to Gemini API
                file_ref = upload_pdf_to_gemini(pdf_path)

                # Extract text with the text_extraction_prompt
                extracted_text = extract_text_with_gemini(file_ref, text_extraction_prompt)

                # Generate summary with the custom_prompt and response_schema
                summary, _, _ = summarize_content_with_gemini(file_ref, custom_prompt, response_schema)

                # Send the summary and extracted text to Airtable
                send_to_airtable(record_id, summary, extracted_text, target_field_id)

        except Exception as e:
            error_message = f"An error occurred during processing: {str(e)}"
            logger.error(error_message)
            logger.error(traceback.format_exc())
            send_to_airtable(record_id, {"error": error_message}, "", target_field_id)

    # Submit the task to the thread pool
    executor.submit(process)

@app.route('/process_pdf/assessment', methods=['POST'])
def process_pdf_assessment_route():
    try:
        data = request.json  # Assuming Airtable sends JSON data
        pdf_url = data.get('pdf_url')
        record_id = data.get('record_id')
        custom_prompt = data.get('custom_prompt')
        response_schema = data.get('response_schema')
        text_extraction_prompt = data.get('text_extraction_prompt')
        target_field_id = data.get('targetFieldId')  # Extract targetFieldId

        # Convert response_schema from string to dictionary if needed
        if isinstance(response_schema, str):
            response_schema = json.loads(response_schema)

        if pdf_url and record_id and response_schema and text_extraction_prompt and target_field_id:
            process_pdf_async_assessment(pdf_url, record_id, custom_prompt, response_schema, text_extraction_prompt, target_field_id)
            return jsonify({"status": "processing started"}), 200
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

@app.route('/process_pdf/submission', methods=['POST'])
def process_pdf_submission_route():
    try:
        data = request.json  # Assuming Airtable sends JSON data
        pdf_url = data.get('pdf_url')
        record_id = data.get('record_id')
        custom_prompt = data.get('custom_prompt')
        response_schema = data.get('response_schema')
        text_extraction_prompt = data.get('text_extraction_prompt')
        target_field_id = data.get('targetFieldId')  # Extract targetFieldId

        # Convert response_schema from string to dictionary if needed
        if isinstance(response_schema, str):
            response_schema = json.loads(response_schema)

        if pdf_url and record_id and response_schema and text_extraction_prompt and target_field_id:
            process_pdf_async_submission(pdf_url, record_id, custom_prompt, response_schema, text_extraction_prompt, target_field_id)
            return jsonify({"status": "processing started"}), 200
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

# Ensure the thread pool executor shuts down cleanly
atexit.register(executor.shutdown)

if __name__ == '__main__':
    app.run(debug=True)

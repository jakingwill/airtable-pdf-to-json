from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
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
executor = ThreadPoolExecutor(max_workers=4)

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

def process_pdf_with_gemini(file_ref, custom_prompt, response_schema):
    """
    Process the uploaded PDF using the Gemini API.
    """
    try:
        model = genai.GenerativeModel(model_name='gemini-1.5-flash')

        if isinstance(response_schema, dict):
            response_schema_str = json.dumps(response_schema, indent=2)
        else:
            response_schema_str = response_schema

        full_prompt = f"{custom_prompt}\n\nPlease extract the information according to the following schema:\n\n{response_schema_str}"
        logger.info(f"Full prompt for summarization: {full_prompt}")

        response = model.generate_content([file_ref, full_prompt])
        logger.info(f"Response from Gemini: {response}")

        if response.candidates and response.candidates[0].content.parts:
            json_response = response.candidates[0].content.parts[0].text
            logger.info(f"Extracted JSON response from Gemini: {json_response}")
            return json_response
        else:
            logger.warning("Content blocked or no content extracted from text file.")
            return {"error": "Content blocked or no content extracted from text file."}
    except Exception as e:
        logger.error(f"Error in process_pdf_with_gemini: {str(e)}")
        raise

def send_to_airtable(record_id, summary, target_field_id):
    """
    Send the processed data to the Airtable webhook.
    """
    try:
        data = {
            "record_id": record_id,
            "summary": summary,
            "target_field_id": target_field_id
        }
        response = requests.post(airtable_webhook_url, json=data)
        response.raise_for_status()
        logger.info("Successfully sent data to Airtable")
    except requests.RequestException as e:
        logger.error(f"Error sending data to Airtable: {str(e)}")
        raise

def process_pdf_async(pdf_url, record_id, custom_prompt, response_schema, target_field_id):
    def process():
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                request_dir = pathlib.Path(temp_dir)

                # Download the PDF
                pdf_path = download_pdf(pdf_url, request_dir)

                # Upload the PDF to Gemini API
                file_ref = upload_pdf_to_gemini(pdf_path)

                # Process the PDF with Gemini and generate a summary
                json_response = process_pdf_with_gemini(file_ref, custom_prompt, response_schema)

                # Send the summary to Airtable
                send_to_airtable(record_id, json_response, target_field_id)

        except Exception as e:
            error_message = f"An error occurred during processing: {str(e)}"
            logger.error(error_message)
            logger.error(traceback.format_exc())
            send_to_airtable(record_id, {"error": error_message}, target_field_id)

    # Submit the task to the thread pool
    executor.submit(process)

@app.route('/process_pdf', methods=['POST'])
def process_pdf_route():
    try:
        data = request.json  # Assuming Airtable sends JSON data
        pdf_url = data.get('pdf_url')
        record_id = data.get('record_id')
        custom_prompt = data.get('custom_prompt')
        response_schema = data.get('response_schema')
        target_field_id = data.get('targetFieldId')  # Extract targetFieldId

        # Convert response_schema from string to dictionary if needed
        if isinstance(response_schema, str):
            response_schema = json.loads(response_schema)

        if pdf_url and record_id and response_schema and target_field_id:
            process_pdf_async(pdf_url, record_id, custom_prompt, response_schema, target_field_id)
            return jsonify({"status": "processing started"}), 200
        else:
            missing_fields = [field for field in ['pdf_url', 'record_id', 'response_schema', 'targetFieldId'] if not locals().get(field)]
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

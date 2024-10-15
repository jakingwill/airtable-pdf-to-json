import os
import json
import logging
import traceback
import tempfile
import pathlib
import requests
import atexit
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, request, jsonify
import google.generativeai as genai

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Retrieve environment variables
gemini_api_key = os.getenv('GEMINI_API_KEY')
airtable_webhook_url = os.getenv('AIRTABLE_WEBHOOK')

# Validate environment variables
if not gemini_api_key:
    logger.error("GEMINI_API_KEY environment variable not set.")
    exit(1)

if not airtable_webhook_url:
    logger.error("AIRTABLE_WEBHOOK environment variable not set.")
    exit(1)

# Configure Gemini API client
genai.configure(api_key=gemini_api_key)

# Create a thread pool executor for tasks
executor = ThreadPoolExecutor(max_workers=1)

def download_pdf(pdf_url, download_folder):
    # [No changes needed here]

def upload_pdf_to_gemini(pdf_path):
    # [No changes needed here]

def extract_text_with_gemini(file_ref, text_extraction_prompt):
    """
    Extract text from the PDF using the Gemini API.
    """
    try:
        # Initialize the model with appropriate generation configuration
        model = genai.GenerativeModel(
            model_name='models/gemini-1.5-flash',  # Updated model name
            generation_config={
                'temperature': 0.7,
                'max_output_tokens': 2048
            }
        )

        # Start a chat with the model
        convo = model.start_chat()

        # Send the text extraction prompt along with the file reference
        response = convo.send_message([file_ref, text_extraction_prompt])

        # Check if response is available
        if response.text:
            extracted_text = response.text
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
    Summarize the content of the PDF using the Gemini API.
    """
    try:
        # Prepare the generation configuration with the response_schema
        generation_config = {
            'temperature': 0.7,
            'max_output_tokens': 2048,
            'response_mime_type': 'application/json',
            'response_schema': response_schema  # Should be a type hint or genai.protos.Schema
        }

        # Initialize the model with the generation configuration
        model = genai.GenerativeModel(
            model_name='models/gemini-1.5-flash',  # Updated model name
            generation_config=generation_config
        )

        # Start a chat with the model using the custom prompt as the system prompt
        convo = model.start_chat(system_prompt=custom_prompt)

        # Send the file reference to the model
        response = convo.send_message([file_ref])

        # Check if response is available
        if response.text:
            try:
                # Parse the JSON-formatted response text
                summary = json.loads(response.text)
                logger.info(f"Parsed summary: {summary}")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse summary as JSON: {str(e)}")
                summary = response.text  # Fallback to raw text
        else:
            logger.warning("No summary extracted from the PDF.")
            summary = ""
        return summary
    except Exception as e:
        logger.error(f"Error in summarize_content_with_gemini: {str(e)}")
        raise

def send_to_airtable(record_id, summary, extracted_text, target_field_id):
    # [No changes needed here]

def process_pdf_async(pdf_url, record_id, custom_prompt, response_schema, text_extraction_prompt, target_field_id):
    # [No changes needed here]

@app.route('/process_pdf', methods=['POST'])
def process_pdf_route():
    # [No changes needed here]

# Ensure the thread pool executor shuts down cleanly
atexit.register(executor.shutdown)

if __name__ == '__main__':
    app.run(debug=False)

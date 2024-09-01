from concurrent.futures import ThreadPoolExecutor
import atexit
import os
import google.generativeai as genai
import requests
from flask import Flask, request, jsonify
import uuid
import fitz  # PyMuPDF
import pathlib
import tqdm
import json
import tempfile
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import time
import backoff
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

# Create a thread pool executor with a fixed number of threads
executor = ThreadPoolExecutor(max_workers=10)

@backoff.on_exception(backoff.expo, Exception, max_tries=5)
def download_pdf(pdf_url, download_folder):
    """
    Download a PDF file from the given URL.
    Uses exponential backoff to retry the download on failure.
    """
    try:
        download_folder = pathlib.Path(download_folder)
        response = requests.get(pdf_url, timeout=60)  # 60 seconds timeout
        response.raise_for_status()  # Raises an HTTPError for bad responses
        file_path = download_folder / 'downloaded_pdf.pdf'
        with file_path.open('wb') as file:
            file.write(response.content)
        logger.info(f"Downloaded PDF to: {file_path}")
        return str(file_path)
    except requests.RequestException as e:
        logger.error(f"Error downloading PDF from {pdf_url}: {str(e)}")
        raise

@backoff.on_exception(backoff.expo, Exception, max_tries=5)
def extract_images_from_pdf(pdf_path, output_dir):
    """
    Extract images from a PDF file.
    Uses exponential backoff to retry the extraction on failure.
    """
    image_files = []
    try:
        logger.info(f"Opening PDF: {pdf_path}")
        with fitz.open(pdf_path) as doc:
            logger.info(f"Number of pages in PDF: {len(doc)}")
            for page_num, page in enumerate(doc):
                try:
                    logger.info(f"Processing page {page_num + 1}")
                    pix = page.get_pixmap()
                    image_filename = output_dir / f"image-{page_num + 1}.jpg"
                    pix.save(image_filename)
                    image_files.append(str(image_filename))
                    logger.info(f"Saved image for page {page_num + 1}: {image_filename}")
                except Exception as e:
                    logger.error(f"Error processing page {page_num + 1}: {str(e)}")
    except Exception as e:
        logger.error(f"Error opening PDF: {str(e)}")
        raise
    return image_files

@backoff.on_exception(backoff.expo, Exception, max_tries=5)
def upload_to_gemini(file_path):
    """
    Upload a file to Gemini.
    Uses exponential backoff to retry the upload on failure.
    """
    try:
        file = genai.upload_file(path=str(file_path), display_name=f"Extracted Text")
        logger.info(f"Successfully uploaded {file_path} to Gemini")
        return file
    except Exception as e:
        logger.error(f"Error in upload_to_gemini: {str(e)}")
        raise

@backoff.on_exception(backoff.expo, Exception, max_tries=5)
def extract_text_from_images(image_files, text_extraction_prompt):
    """
    Extract text from a list of images using the Gemini API.
    Uses exponential backoff to retry the text extraction on failure.
    """
    model = genai.GenerativeModel(model_name='models/gemini-1.5-flash')

    extracted_text = ""
    for img in tqdm.tqdm(image_files):
        try:
            file = genai.upload_file(path=str(img), display_name=f"Page {pathlib.Path(img).stem}")
            prompt = [text_extraction_prompt] + [file] + ["[END]\n\nPlease extract the text from these images."]

            safety_settings = {
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }

            response = model.generate_content(prompt, safety_settings=safety_settings)
            if response.prompt_feedback.block_reason == 0 and response.candidates and response.candidates[0].content.parts:
                extracted_text += response.candidates[0].content.parts[0].text + "\n\n"
            else:
                logger.warning(f"Content blocked or no content extracted from image: {img}")
        except Exception as e:
            logger.error(f"Error extracting text from image {img}: {str(e)}")

    logger.info(f"Extracted text from images: {extracted_text[:500]}")
    return extracted_text

@backoff.on_exception(backoff.expo, Exception, max_tries=5)
def summarize_content(extracted_text_file, custom_prompt, response_schema):
    """
    Summarize the content of a text file using the Gemini API.
    Uses exponential backoff to retry the summarization on failure.
    """
    try:
        if isinstance(response_schema, dict):
            response_schema_str = json.dumps(response_schema, indent=2)
        else:
            response_schema_str = response_schema

        full_prompt = f"{custom_prompt}\n\nPlease extract the information according to the following schema:\n\n{response_schema_str}"
        logger.info(f"Full prompt for summarization: {full_prompt}")

        model = genai.GenerativeModel(model_name='models/gemini-1.5-flash')
        prompt = [full_prompt] + [extracted_text_file] + ["[END]\n\nPlease extract the text according to the schema."]

        safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        response = model.generate_content(prompt, safety_settings=safety_settings)
        logger.info(f"Response from Gemini: {response}")
        if response.prompt_feedback.block_reason == 0 and response.candidates and response.candidates[0].content.parts:
            json_response = response.candidates[0].content.parts[0].text
            logger.info(f"Extracted JSON response from Gemini: {json_response}")
            return json_response
        else:
            logger.warning("Content blocked or no content extracted from text file.")
            return {"error": "Content blocked or no content extracted from text file."}
    except Exception as e:
        logger.error(f"Error in summarize_content: {str(e)}")
        raise

@backoff.on_exception(backoff.expo, Exception, max_tries=5)
def send_to_airtable(record_id, summary, extracted_text, target_field_id):
    """
    Send the processed data to the Airtable webhook.
    Uses exponential backoff to retry the sending on failure.
    """
    try:
        webhook_url = airtable_webhook_url
        logger.info(f"Webhook URL: {webhook_url}")
        if not webhook_url.startswith("https://"):
            if webhook_url.startswith("http://"):
                webhook_url = webhook_url.replace("http://", "https://")
            else:
                webhook_url = f"https://{webhook_url}"

        data = {
            "record_id": record_id,
            "summary": summary,
            "extracted_text": extracted_text,
            "target_field_id": target_field_id
        }
        response = requests.post(webhook_url, json=data)
        response.raise_for_status()
        logger.info("Successfully sent data to Airtable")
    except requests.RequestException as e:
        logger.error(f"Error sending data to Airtable: {str(e)}")
        raise

def process_pdf_async(pdf_url, record_id, custom_prompt, text_extraction_prompt, response_schema, target_field_id):
    def process():
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                request_dir = pathlib.Path(temp_dir)

                pdf_path = download_pdf(pdf_url, request_dir)
                output_dir = request_dir / 'output'
                output_dir.mkdir(exist_ok=True)

                # Extract images from PDF
                image_files = extract_images_from_pdf(pdf_path, output_dir)

                # Extract text from images using Gemini
                full_text = extract_text_from_images(image_files, text_extraction_prompt)

                if full_text.strip():
                    text_file_path = request_dir / 'extracted_text.txt'
                    with text_file_path.open('w') as text_file:
                        text_file.write(full_text)

                    uploaded_file = upload_to_gemini(text_file_path)

                    json_response = summarize_content(uploaded_file, custom_prompt, response_schema)
                    logger.info(f"JSON response to be sent to Airtable: {json_response}")

                    # Read the extracted text file
                    with open(text_file_path, 'r') as f:
                        extracted_text = f.read()

                    # Send the data to Airtable
                    send_to_airtable(record_id, json_response, extracted_text, target_field_id)
                else:
                    error_message = "Extraction failed. No text could be extracted from the PDF images."
                    logger.error(error_message)
                    send_to_airtable(record_id, {"error": error_message}, "No text extracted", target_field_id)

        except Exception as e:
            error_message = f"An error occurred during processing: {str(e)}"
            logger.error(error_message)
            logger.error(traceback.format_exc())
            send_to_airtable(record_id, {"error": error_message}, "An error occurred", target_field_id)

    # Submit the task to the thread pool
    executor.submit(process)

@app.route('/process_pdf', methods=['POST'])
def process_pdf_route():
    try:
        data = request.json  # Assuming Airtable sends JSON data
        pdf_url = data.get('pdf_url')
        record_id = data.get('record_id')
        custom_prompt = data.get('custom_prompt')
        text_extraction_prompt = data.get('text_extraction_prompt')
        response_schema = data.get('response_schema')
        target_field_id = data.get('targetFieldId')  # Extract targetFieldId

        # Convert response_schema from string to dictionary if needed
        if isinstance(response_schema, str):
            response_schema = json.loads(response_schema)

        if pdf_url and record_id and response_schema and text_extraction_prompt and target_field_id:
            process_pdf_async(pdf_url, record_id, custom_prompt, text_extraction_prompt, response_schema, target_field_id)
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

if __name__ == "__main__":
    app.run(debug=True)

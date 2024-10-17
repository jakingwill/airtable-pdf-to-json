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
from PIL import Image

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

def is_image_file(file_path):
    """
    Check if the file is an image based on its extension.
    """
    image_extensions = ['.jpg', '.jpeg', '.png', '.heic', '.tiff', '.bmp']
    return pathlib.Path(file_path).suffix.lower() in image_extensions

def convert_image_to_pdf(image_path, output_folder):
    """
    Convert an image file to a PDF and save it in the output folder.
    """
    try:
        image = Image.open(image_path)
        pdf_path = pathlib.Path(output_folder) / 'converted_image.pdf'
        # Convert the image to RGB mode (required for PDF conversion)
        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB")
        image.save(pdf_path, "PDF", resolution=100.0)
        logger.info(f"Converted image to PDF: {pdf_path}")
        return str(pdf_path)
    except Exception as e:
        logger.error(f"Error converting image to PDF: {str(e)}")
        raise

def process_pdf_or_image(pdf_or_image_url, download_folder):
    """
    Download the file and process it, converting images to PDF if necessary.
    """
    try:
        download_folder = pathlib.Path(download_folder)
        download_folder.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
        file_path = download_folder / 'downloaded_file'

        response = requests.get(pdf_or_image_url, timeout=60)  # Set a timeout for the request
        response.raise_for_status()  # Raise an error if the request failed

        file_extension = pathlib.Path(pdf_or_image_url).suffix
        file_path = file_path.with_suffix(file_extension)

        with open(file_path, 'wb') as file:
            file.write(response.content)

        logger.info(f"Downloaded file to: {file_path}")

        # Check if it's an image, and convert it to PDF if needed
        if is_image_file(file_path):
            file_path = convert_image_to_pdf(file_path, download_folder)

        return str(file_path)
    except requests.RequestException as e:
        logger.error(f"Error downloading file from {pdf_or_image_url}: {str(e)}")
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
    Summarize the content of the PDF and determine the assessment type and assessment name using the Gemini API.
    """
    try:
        model = genai.GenerativeModel(model_name='gemini-1.5-flash')

        # Include prompt to determine assessment type and assessment name
        assessment_type_prompt = (
            "Your job is to analyse the extracted text and decide which of the following options this assessment meets, in terms of assessment type. "
            "Essay: You should select this option if you think the assessment is an essay-style assessment that probably contains only one question. "
            "Exam style: You should select this option if you think the assessment is an exam-style assessment. "
            "The definition of this is that it contains multiple questions that the student has to answer, and likely has question numbering (e.g., 1.1, 1.2, etc.). "
            "The output that you give me should simply be the assessment type that you think is most likely. "
            "Therefore your output should simply either be: 'Essay' OR 'Exam style'."
        )

        assessment_name_prompt = (
            "Please give this assessment an appropriate name. The title of the assessment is often found in the beginning of the extracted text of the assessment so use that, "
            "but if not, provide an intuitive name for the assessment using the content provided. "
            "If you can identify the subject and the grade in the text provided, please include that in the name. "
            "In your output, please only provide the name of the assessment - no pre-text or post-text should exist in your output."
        )

        full_prompt = f"{custom_prompt}\n\nPlease extract the information according to the following schema:\n\n{json.dumps(response_schema, indent=2)}\n\n{assessment_type_prompt}\n\n{assessment_name_prompt}"
        logger.info(f"Full prompt for summarization: {full_prompt}")

        # Generate content using the custom prompt and generation config
        response = model.generate_content([file_ref, full_prompt])
        logger.info(f"Response from Gemini: {response}")

        if response.candidates and response.candidates[0].content.parts:
            summary = response.candidates[0].content.parts[0].text
            logger.info(f"Extracted summary from PDF: {summary}")

            # Determine assessment type (Look for 'Essay' or 'Exam style' in the response)
            if "Essay" in summary:
                assessment_type = "Essay"
            elif "Exam style" in summary:
                assessment_type = "Exam style"
            else:
                assessment_type = "Exam style"  # Fallback to 'Exam style' if neither is detected

            # Extract assessment name from the summary (Assume it follows the assessment type)
            assessment_name = summary.splitlines()[-1].strip() if summary else "Unknown"

            logger.info(f"Determined assessment type: {assessment_type}")
            logger.info(f"Determined assessment name: {assessment_name}")
            return summary, assessment_type, assessment_name
        else:
            logger.warning("No summary extracted from the PDF.")
            return "", "Exam style", "Unknown"
    except Exception as e:
        logger.error(f"Error in summarize_content_with_gemini: {str(e)}")
        raise

def send_to_airtable(record_id, summary, extracted_text, target_field_id, assessment_type=None, assessment_name=None):
    """
    Send the processed data, assessment type, and assessment name to the Airtable webhook.
    """
    try:
        data = {
            "record_id": record_id,
            "summary": summary,  # JSON-formatted summary
            "extracted_text": extracted_text,  # Plain text extracted from the PDF
            "target_field_id": target_field_id
        }
        if assessment_type:
            data["assessmentType"] = assessment_type  # Include the assessment type if provided
        if assessment_name:
            data["assessmentName"] = assessment_name  # Include the assessment name if provided

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

                # Download the PDF or image and convert it to PDF if necessary
                pdf_path = process_pdf_or_image(pdf_url, request_dir)

                # Upload the PDF to Gemini API
                file_ref = upload_pdf_to_gemini(pdf_path)

                # Extract text with the text_extraction_prompt
                extracted_text = extract_text_with_gemini(file_ref, text_extraction_prompt)

                # Generate summary with the custom_prompt and response_schema, and determine the assessment type and name
                summary, assessment_type, assessment_name = summarize_content_with_gemini(file_ref, custom_prompt, response_schema)

                # Send the summary, extracted text, assessment type, and assessment name to Airtable
                send_to_airtable(record_id, summary, extracted_text, target_field_id, assessment_type, assessment_name)

        except Exception as e:
            error_message = f"An error occurred during processing: {str(e)}"
            logger.error(error_message)
            logger.error(traceback.format_exc())
            send_to_airtable(record_id, {"error": error_message}, "", target_field_id)

    # Submit the task to the thread pool
    executor.submit(process)

def process_pdf_async_submission(pdf_url, record_id, custom_prompt, response_schema, text_extraction_prompt, target_field_id):
    def process():
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                request_dir = pathlib.Path(temp_dir)

                # Download the PDF or image and convert it to PDF if necessary
                pdf_path = process_pdf_or_image(pdf_url, request_dir)

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

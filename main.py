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
from json_repair import repair_json
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

def validate_request_data(data):
    """Validate incoming request data and return detailed error messages."""
    required_fields = {
        'pdf_url': str,
        'record_id': str,
        'custom_prompt': str,
        'response_schema': (dict, str),
        'text_extraction_prompt': str,
        'targetFieldId': str
    }
    
    errors = []
    
    for field, expected_type in required_fields.items():
        value = data.get(field)
        if value is None or value == "":
            errors.append(f"Missing or empty required field: {field}")
        elif not isinstance(value, expected_type) and not (isinstance(expected_type, tuple) and any(isinstance(value, t) for t in expected_type)):
            errors.append(f"Invalid type for {field}: expected {expected_type}, got {type(value)}")
            
    # Additional validation for non-empty prompts
    if data.get('custom_prompt') and len(data.get('custom_prompt').strip()) == 0:
        errors.append("custom_prompt is empty after stripping whitespace")
    if data.get('text_extraction_prompt') and len(data.get('text_extraction_prompt').strip()) == 0:
        errors.append("text_extraction_prompt is empty after stripping whitespace")
            
    return errors

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

def extract_text_with_gemini(file_ref, text_extraction_prompt, temperature=0):
    try:
        if not text_extraction_prompt or not text_extraction_prompt.strip():
            raise ValueError("Text extraction prompt cannot be empty")

        model = genai.GenerativeModel(model_name='gemini-1.5-flash')
        temperature = float(temperature)
        generation_config = genai.types.GenerationConfig(temperature=temperature)
        
        logger.info(f"Extracting text with prompt: {text_extraction_prompt[:100]}...")
        response = model.generate_content([file_ref, text_extraction_prompt], generation_config=generation_config)
        
        if response.candidates and response.candidates[0].content.parts:
            extracted_text = response.candidates[0].content.parts[0].text
            logger.info(f"Successfully extracted text (length: {len(extracted_text)})")
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

def summarize_content_with_gemini(file_ref, custom_prompt, response_schema, assessment_type_prompt="", assessment_name_prompt="", marking_guide_prompt="", temperature=0):
    try:
        if not custom_prompt or not custom_prompt.strip():
            raise ValueError("Custom prompt cannot be empty")

        model = genai.GenerativeModel(model_name='gemini-1.5-flash')
        temperature = float(temperature)
        generation_config = genai.types.GenerationConfig(temperature=temperature)

        logger.info(f"Generating content with prompt length: {len(custom_prompt)}")
        json_response = model.generate_content([file_ref, custom_prompt], generation_config=generation_config)

        if json_response.candidates and json_response.candidates[0].content.parts:
            raw_json_content = json_response.candidates[0].content.parts[0].text
            json_content = validate_and_repair_json(raw_json_content)
            logger.info("Successfully generated and validated JSON content")
        else:
            logger.warning("No JSON content generated.")
            json_content = ""

        return json_content, "", "", ""

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

        logger.info(f"Sending data to Airtable for record: {record_id}")
        response = requests.post(airtable_webhook_url, json=data)
        response.raise_for_status()
        logger.info("Successfully sent data to Airtable")
    except requests.RequestException as e:
        logger.error(f"Error sending data to Airtable: {str(e)}")
        raise

def process_pdf_async_submission(pdf_url, record_id, custom_prompt, response_schema, text_extraction_prompt, target_field_id, temperature=0):
    def process():
        try:
            # Log incoming parameters
            logger.info(f"""
Processing submission with parameters:
PDF URL: {pdf_url}
Record ID: {record_id}
Custom Prompt Length: {len(custom_prompt) if custom_prompt else 0}
Text Extraction Prompt Length: {len(text_extraction_prompt) if text_extraction_prompt else 0}
Temperature: {temperature}
            """)

            # Validate prompts
            if not custom_prompt or not custom_prompt.strip():
                raise ValueError("Custom prompt is empty or contains only whitespace")
            if not text_extraction_prompt or not text_extraction_prompt.strip():
                raise ValueError("Text extraction prompt is empty or contains only whitespace")

            with tempfile.TemporaryDirectory() as temp_dir:
                request_dir = pathlib.Path(temp_dir)

                # Download and process PDF
                pdf_path = download_pdf(pdf_url, request_dir)
                file_ref = upload_pdf_to_gemini(pdf_path)
                
                # Extract text
                logger.info("Attempting text extraction...")
                extracted_text = extract_text_with_gemini(file_ref, text_extraction_prompt, temperature)
                
                if not extracted_text:
                    raise ValueError("No text extracted from the PDF. Cannot proceed with JSON generation.")

                logger.info(f"Successfully extracted text of length: {len(extracted_text)}")

                # Generate JSON content
                updated_custom_prompt = f"{custom_prompt}\n\nExtracted Text:\n{extracted_text}\n\nSchema:\n{json.dumps(response_schema, indent=2)}"
                logger.info("Generating JSON content...")
                
                json_content, _, _, _ = summarize_content_with_gemini(
                    file_ref, 
                    updated_custom_prompt, 
                    response_schema, 
                    "", 
                    "", 
                    "", 
                    temperature
                )

                # Send results to Airtable
                send_to_airtable(
                    record_id,
                    json_content,
                    "",
                    "",
                    extracted_text,
                    "",
                    target_field_id,
                    "Successfully processed submission by Gemini"
                )

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
        logger.info("Received request data: %s", json.dumps(data, indent=2))
        
        # Validate request data
        validation_errors = validate_request_data(data)
        if validation_errors:
            error_message = f"Validation errors: {', '.join(validation_errors)}"
            logger.error(error_message)
            return jsonify({"error": error_message}), 400

        pdf_url = data.get('pdf_url')
        record_id = data.get('record_id')
        custom_prompt = data.get('custom_prompt', '').strip()
        response_schema = data.get('response_schema')
        text_extraction_prompt = data.get('text_extraction_prompt', '').strip()
        target_field_id = data.get('targetFieldId')
        temperature = data.get('temperature', 0)

        if isinstance(response_schema, str):
            try:
                response_schema = json.loads(response_schema)
            except JSONDecodeError as e:
                return jsonify({"error": f"Invalid response_schema JSON: {str(e)}"}), 400

        process_pdf_async_submission(
            pdf_url, 
            record_id, 
            custom_prompt, 
            response_schema, 
            text_extraction_prompt, 
            target_field_id, 
            temperature
        )
        
        return jsonify({"status": "submission processing started"}), 200
        
    except json.JSONDecodeError as e:
        error_message = f"Invalid JSON format in request body: {str(e)}"
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

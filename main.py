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

def generate_marking_guide_with_gemini(file_ref, marking_guide_prompt, temperature=0):
    try:
        model = genai.GenerativeModel(model_name='gemini-1.5-flash')
        # Convert temperature to float
        temperature = float(temperature)
        generation_config = genai.types.GenerationConfig(temperature=temperature)
        response = model.generate_content([file_ref, marking_guide_prompt], generation_config=generation_config)
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

def summarize_content_with_gemini(file_ref, custom_prompt, response_schema, assessment_type_prompt=None, assessment_name_prompt=None, marking_guide_prompt=None, temperature=0):
    try:
        model = genai.GenerativeModel(model_name='gemini-1.5-flash')
        temperature = float(temperature)
        generation_config = genai.types.GenerationConfig(temperature=temperature)

        # Optional: Generate the marking guide if prompt is provided
        marking_guide = ""
        if marking_guide_prompt:
            marking_guide = generate_marking_guide_with_gemini(file_ref, marking_guide_prompt, temperature)
            if not marking_guide:
                logger.warning("No marking guide generated.")
        
        # Prepare the custom prompt for JSON extraction
        json_prompt = f"{custom_prompt}\n\nSchema:\n{json.dumps(response_schema, indent=2)}"
        if marking_guide:
            json_prompt += f"\n\nUse the following marking guide:\n\n{marking_guide}"

        # Generate JSON content
        json_response = model.generate_content([file_ref, json_prompt], generation_config=generation_config)
        if json_response.candidates and json_response.candidates[0].content.parts:
            raw_json_content = json_response.candidates[0].content.parts[0].text
            json_content = validate_and_repair_json(raw_json_content)
            logger.info(f"Validated and repaired JSON content: {json_content}")
        else:
            json_content = ""

        # Optional: Generate assessment type and name if prompts are provided
        assessment_type = ""
        assessment_name = ""
        if assessment_type_prompt:
            type_response = model.generate_content([file_ref, assessment_type_prompt], generation_config=generation_config)
            if type_response.candidates and type_response.candidates[0].content.parts:
                assessment_type = type_response.candidates[0].content.parts[0].text.strip()
        
        if assessment_name_prompt:
            name_response = model.generate_content([file_ref, assessment_name_prompt], generation_config=generation_config)
            if name_response.candidates and name_response.candidates[0].content.parts:
                assessment_name = name_response.candidates[0].content.parts[0].text.strip()

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
            logger.info(f"Processing submission with parameters: PDF URL: {pdf_url}, Record ID: {record_id}")

            with tempfile.TemporaryDirectory() as temp_dir:
                pdf_path = download_pdf(pdf_url, temp_dir)
                file_ref = upload_pdf_to_gemini(pdf_path)

                extracted_text = extract_text_with_gemini(file_ref, text_extraction_prompt, temperature)
                if not extracted_text:
                    raise ValueError("No text extracted from the PDF. Cannot proceed.")

                json_content, _, _, _ = summarize_content_with_gemini(
                    file_ref, 
                    custom_prompt, 
                    response_schema, 
                    temperature=temperature
                )

                send_to_airtable(
                    record_id,
                    json_content,
                    "",  # No assessment type for submission
                    "",  # No assessment name for submission
                    extracted_text,
                    "",  # No marking guide for submission
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

@app.route('/process_pdf/assessment', methods=['POST'])
def process_pdf_assessment_route():
    try:
        data = request.json
        logger.info("Received request data for assessment: %s", json.dumps(data, indent=2))
        
        # Validate required fields
        required_fields = ['pdf_url', 'record_id', 'custom_prompt', 'response_schema', 
                           'text_extraction_prompt', 'targetFieldId', 'assessment_type_prompt', 
                           'assessment_name_prompt', 'marking_guide_prompt']
        missing_fields = [field for field in required_fields if not data.get(field)]
        if missing_fields:
            error_message = f"Missing required fields: {', '.join(missing_fields)}"
            logger.error(error_message)
            return jsonify({"error": error_message}), 400
        
        pdf_url = data.get('pdf_url')
        record_id = data.get('record_id')
        custom_prompt = data.get('custom_prompt', '').strip()
        response_schema = data.get('response_schema')
        text_extraction_prompt = data.get('text_extraction_prompt', '').strip()
        target_field_id = data.get('targetFieldId')
        assessment_type_prompt = data.get('assessment_type_prompt', '').strip()
        assessment_name_prompt = data.get('assessment_name_prompt', '').strip()
        marking_guide_prompt = data.get('marking_guide_prompt', '').strip()
        temperature = data.get('temperature', 0)

        # Parse response_schema if provided as string
        if isinstance(response_schema, str):
            try:
                response_schema = json.loads(response_schema)
            except JSONDecodeError as e:
                error_message = f"Invalid response_schema JSON: {str(e)}"
                logger.error(error_message)
                return jsonify({"error": error_message}), 400

        process_pdf_async_assessment(
            pdf_url, 
            record_id, 
            custom_prompt, 
            response_schema, 
            text_extraction_prompt, 
            target_field_id, 
            assessment_type_prompt, 
            assessment_name_prompt, 
            marking_guide_prompt, 
            temperature
        )
        
        return jsonify({"status": "assessment processing started"}), 200
        
    except json.JSONDecodeError as e:
        error_message = f"Invalid JSON format in request body: {str(e)}"
        logger.error(error_message)
        return jsonify({"error": error_message}), 400
    except Exception as e:
        error_message = f"An unexpected error occurred: {str(e)}"
        logger.error(error_message)
        logger.error(traceback.format_exc())
        return jsonify({"error": error_message}), 500

def process_pdf_async_assessment(pdf_url, record_id, custom_prompt, response_schema, text_extraction_prompt, 
                                  target_field_id, assessment_type_prompt, assessment_name_prompt, 
                                  marking_guide_prompt, temperature=0):
    def process():
        try:
            logger.info(f"""
Processing assessment with parameters:
PDF URL: {pdf_url}
Record ID: {record_id}
Custom Prompt Length: {len(custom_prompt) if custom_prompt else 0}
Text Extraction Prompt Length: {len(text_extraction_prompt) if text_extraction_prompt else 0}
Temperature: {temperature}
            """)

            with tempfile.TemporaryDirectory() as temp_dir:
                request_dir = pathlib.Path(temp_dir)

                # Download and process PDF
                pdf_path = download_pdf(pdf_url, request_dir)
                file_ref = upload_pdf_to_gemini(pdf_path)
                
                # Extract text
                logger.info("Attempting text extraction...")
                extracted_text = extract_text_with_gemini(file_ref, text_extraction_prompt, temperature)
                
                if not extracted_text:
                    raise ValueError("No text extracted from the PDF. Cannot proceed with processing.")

                logger.info(f"Successfully extracted text of length: {len(extracted_text)}")

                # Generate JSON content and additional details
                json_content, assessment_type, assessment_name, new_marking_guide = summarize_content_with_gemini(
                    file_ref, 
                    custom_prompt, 
                    response_schema, 
                    assessment_type_prompt, 
                    assessment_name_prompt, 
                    marking_guide_prompt, 
                    temperature
                )

                # Send results to Airtable
                send_to_airtable(
                    record_id,
                    json_content,
                    assessment_type,
                    assessment_name,
                    extracted_text,
                    new_marking_guide,
                    target_field_id,
                    "Successfully processed assessment by Gemini"
                )

        except Exception as e:
            error_message = f"An error occurred during assessment processing: {str(e)}"
            logger.error(error_message)
            logger.error(traceback.format_exc())
            send_to_airtable(record_id, "", "", "", "", "", target_field_id, error_message)

    executor.submit(process)


if __name__ == '__main__':
    app.run(debug=True)

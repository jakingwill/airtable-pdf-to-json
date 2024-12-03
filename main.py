from concurrent.futures import ThreadPoolExecutor
import atexit
import os
import google.generativeai as genai
import requests
from flask import Flask, request, jsonify
import pathlib
import json
tempfile
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
        'combined_prompt': str,
        'response_schema': (dict, str),
        'targetFieldId': str
    }
    
    errors = []
    
    for field, expected_type in required_fields.items():
        value = data.get(field)
        if value is None or value == "":
            errors.append(f"Missing or empty required field: {field}")
        elif not isinstance(value, expected_type) and not (isinstance(expected_type, tuple) and any(isinstance(value, t) for t in expected_type)):
            errors.append(f"Invalid type for {field}: expected {expected_type}, got {type(value)}")
            
    # Additional validation for non-empty combined prompt
    if data.get('combined_prompt') and len(data.get('combined_prompt').strip()) == 0:
        errors.append("combined_prompt is empty after stripping whitespace")
            
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

def summarize_content_with_gemini(file_ref, combined_prompt, temperature=0):
    try:
        model = genai.GenerativeModel(model_name='gemini-1.5-flash')
        temperature = float(temperature)
        generation_config = genai.types.GenerationConfig(temperature=temperature)

        # Generate combined content
        logger.info(f"Extracting information with combined prompt: {combined_prompt[:100]}...")
        response = model.generate_content([file_ref, combined_prompt], generation_config=generation_config)

        if response.candidates and response.candidates[0].content.parts:
            raw_json_content = response.candidates[0].content.parts[0].text
            json_content = validate_and_repair_json(raw_json_content)
            logger.info(f"Validated and repaired JSON content: {json_content}")
            parsed_response = json.loads(json_content)

            subject = parsed_response.get('subject', 'Subject not determined')
            topic = parsed_response.get('topic', 'Topic not determined')
            grade = parsed_response.get('grade', 'Grade not determined')
            curriculum = parsed_response.get('curriculum', 'Curriculum not determined')
            assessment_type = parsed_response.get('assessment_type', 'Assessment type not determined')

            return subject, topic, grade, curriculum, assessment_type

        else:
            logger.warning("No information extracted from the PDF.")
            return "Subject not determined", "Topic not determined", "Grade not determined", "Curriculum not determined", "Assessment type not determined"

    except Exception as e:
        logger.error(f"Error in summarize_content_with_gemini: {str(e)}")
        raise

def send_to_airtable(record_id, json_content, assessment_type, extracted_text, target_field_id, status_message, subject=None, topic=None, grade=None, curriculum=None):
    try:
        # Create the payload
        data = {
            "record_id": record_id,
            "json_content": json_content,
            "assessmentType": assessment_type,
            "extracted_text": extracted_text,
            "status_message": status_message,
            "target_field_id": target_field_id,
            "subject": subject,
            "topic": topic,
            "grade": grade,
            "curriculum": curriculum
        }

        # Log the size of each field
        logger.info("Logging field sizes (in bytes):")
        for key, value in data.items():
            if isinstance(value, str):  # Calculate size only for strings
                logger.info(f"  {key}: {len(value.encode('utf-8'))} bytes")

        # Log total payload size
        payload_size = len(json.dumps(data).encode('utf-8'))
        logger.info(f"Total payload size: {payload_size} bytes")

        # Send the data to Airtable
        logger.info(f"Sending data to Airtable for record: {record_id}")
        response = requests.post(airtable_webhook_url, json=data)
        response.raise_for_status()
        logger.info("Successfully sent data to Airtable")

    except requests.RequestException as e:
        logger.error(f"Error sending data to Airtable: {str(e)}")
        raise

@app.route('/process_pdf/assessment', methods=['POST'])
def process_pdf_assessment_route():
    try:
        data = request.json
        logger.info("Received request data for assessment: %s", json.dumps(data, indent=2))
        
        # Validate required fields
        validation_errors = validate_request_data(data)
        if validation_errors:
            error_message = f"Validation errors: {', '.join(validation_errors)}"
            logger.error(error_message)
            return jsonify({"error": error_message}), 400
        
        pdf_url = data.get('pdf_url')
        record_id = data.get('record_id')
        combined_prompt = data.get('combined_prompt', '').strip()
        response_schema = data.get('response_schema')
        target_field_id = data.get('targetFieldId')
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
            combined_prompt, 
            response_schema, 
            target_field_id, 
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

def process_pdf_async_assessment(pdf_url, record_id, combined_prompt, response_schema, target_field_id, temperature=0):
    def process():
        try:
            logger.info(f"""
Processing assessment with parameters:
PDF URL: {pdf_url}
Record ID: {record_id}
Combined Prompt Length: {len(combined_prompt) if combined_prompt else 0}
Temperature: {temperature}
            """)

            with tempfile.TemporaryDirectory() as temp_dir:
                request_dir = pathlib.Path(temp_dir)

                # Download and process PDF
                pdf_path = download_pdf(pdf_url, request_dir)
                file_ref = upload_pdf_to_gemini(pdf_path)
                
                # Generate JSON content and additional details
                subject, topic, grade, curriculum, assessment_type = summarize_content_with_gemini(
                    file_ref, 
                    combined_prompt, 
                    temperature
                )

                # Send results to Airtable
                send_to_airtable(
                    record_id,
                    "",  # JSON content already extracted separately
                    assessment_type,
                    "",  # No text extraction needed here
                    target_field_id,
                    "Successfully processed assessment by Gemini",
                    subject=subject,
                    topic=topic,
                    grade=grade,
                    curriculum=curriculum
                )

        except Exception as e:
            error_message = f"An error occurred during assessment processing: {str(e)}"
            logger.error(error_message)
            logger.error(traceback.format_exc())
            send_to_airtable(record_id, "", "", "", target_field_id, error_message)

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
        student_name_prompt = data.get('student_name_prompt', '').strip()
        target_field_id = data.get('targetFieldId')
        temperature = data.get('temperature', 0)

        if isinstance(response_schema, str):
            try:
                response_schema = json.loads(response_schema)
            except JSONDecodeError as e:
                return jsonify({"error": f"Invalid response_schema JSON: {str(e)}"}), 400

        # Pass student_name_prompt to the async function
        process_pdf_async_submission(
            pdf_url, 
            record_id, 
            custom_prompt, 
            response_schema, 
            text_extraction_prompt, 
            student_name_prompt, 
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

def process_pdf_async_submission(pdf_url, record_id, custom_prompt, response_schema, 
                                  text_extraction_prompt, student_name_prompt, target_field_id, temperature=0):
    def process():
        try:
            logger.info(f"Processing submission with parameters: PDF URL: {pdf_url}, Record ID: {record_id}")

            with tempfile.TemporaryDirectory() as temp_dir:
                pdf_path = download_pdf(pdf_url, temp_dir)
                file_ref = upload_pdf_to_gemini(pdf_path)

                # Extract text
                extracted_text = extract_text_with_gemini(file_ref, text_extraction_prompt, temperature)
                if not extracted_text:
                    raise ValueError("No text extracted from the PDF. Cannot proceed.")

                # Extract student name using the prompt from Airtable
                if not student_name_prompt or not student_name_prompt.strip():
                    raise ValueError("Student name prompt cannot be empty.")
                student_name = extract_student_name_with_gemini(file_ref, student_name_prompt, temperature)

                # Summarize content and generate JSON
                json_content, _, _, _ = summarize_content_with_gemini(
                    file_ref, 
                    custom_prompt, 
                    response_schema, 
                    temperature=temperature
                )

                # Send results to Airtable
                send_to_airtable(
                    record_id,
                    json_content,
                    "",  # No assessment type for submission
                    "",  # No assessment name for submission
                    extracted_text,
                    "",  # No marking guide for submission
                    target_field_id,
                    f"Successfully processed submission by Gemini. Student name: {student_name}",
                    student_name=student_name  # Pass the extracted student name
                )

        except Exception as e:
            error_message = f"An error occurred during submission processing: {str(e)}"
            logger.error(error_message)
            logger.error(traceback.format_exc())
            send_to_airtable(record_id, "", "", "", "", "", target_field_id, error_message)

    executor.submit(process)

if __name__ == '__main__':
    app.run(debug=True)

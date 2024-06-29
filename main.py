import os
import google.generativeai as genai
import requests
from flask import Flask, request, jsonify
from threading import Thread
import uuid
import fitz  # PyMuPDF
import pathlib
import tqdm
import json
import tempfile

app = Flask(__name__)

# Initialize Gemini API client with your API key
gemini_api_key = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=gemini_api_key)
airtable_webhook_url = os.environ['AIRTABLE_WEBHOOK']

def extract_pdf_content(pdf_path, output_dir):
    full_text = ""
    image_files = []
    with fitz.open(pdf_path) as doc:
        for page_num, page in enumerate(doc):
            page_text = page.get_text().strip()
            if page_text:
                full_text += page_text + "\n\n"
            else:
                # If no text extracted, save as image
                pix = page.get_pixmap()
                image_filename = output_dir / f"image-{page_num + 1}.jpg"
                pix.save(image_filename)
                image_files.append(str(image_filename))

    return full_text, image_files

def download_pdf(pdf_url, download_folder):
    download_folder = pathlib.Path(download_folder)
    response = requests.get(pdf_url)
    if response.status_code == 200:
        file_path = download_folder / 'downloaded_pdf.pdf'
        with file_path.open('wb') as file:
            file.write(response.content)
        return str(file_path)
    else:
        raise Exception(f"Failed to download PDF, status code: {response.status_code}")

def upload_to_gemini(image_files):
    files = []
    for img in tqdm.tqdm(image_files):
        file = genai.upload_file(path=str(img), display_name=f"Page {pathlib.Path(img).stem}")
        files.append(file)
        print(f"Uploaded {img}")
    return files

def summarize_content(files, custom_prompt, response_schema):
    # First, extract text from images
    text_extraction_prompt = "Extract and transcribe all visible text from these images, preserving formatting and structure as much as possible."
    model = genai.GenerativeModel(model_name='models/gemini-1.5-flash')

    text_extraction_response = model.generate_content([text_extraction_prompt] + files)
    extracted_text = text_extraction_response.text

    # Now, use the extracted text to generate the structured response
    full_prompt = f"{custom_prompt}\n\nHere's the extracted text from the PDF:\n\n{extracted_text}\n\nPlease extract the information according to the following schema:"

    generation_config = genai.GenerationConfig(
        response_mime_type='application/json',
        response_schema=response_schema
    )

    response = model.generate_content(full_prompt, generation_config=generation_config)
    json_response = response.candidates[0].content.parts[0].text
    print(f"Extracted JSON response: {json_response}")
    return json_response, extracted_text

def send_to_airtable(record_id, summary):
    webhook_url = airtable_webhook_url
    data = {
        "record_id": record_id,
        "summary": summary
    }
    response = requests.post(webhook_url, json=data)
    if response.status_code == 200:
        print("Successfully sent data to Airtable")
    else:
        print(f"Failed to send data to Airtable: {response.status_code}, {response.text}")

def process_pdf_async(pdf_url, record_id, custom_prompt, response_schema):
    def process():
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                request_dir = pathlib.Path(temp_dir)

                pdf_path = download_pdf(pdf_url, request_dir)
                output_dir = request_dir / 'output'
                output_dir.mkdir(exist_ok=True)

                # Extract content and save as images
                full_text, image_files = extract_pdf_content(pdf_path, output_dir)

                if image_files:
                    files = upload_to_gemini(image_files)
                    if files:
                        json_response, extracted_text = summarize_content(files, custom_prompt, response_schema)

                        # Save extracted text to a file
                        text_file_path = output_dir / 'extracted_text.txt'
                        with open(text_file_path, 'w', encoding='utf-8') as f:
                            f.write(extracted_text)

                        print(f"Extracted JSON response: {json_response}")
                        send_to_airtable(record_id, json_response)
                    else:
                        error_message = "No files uploaded to Gemini."
                        print(error_message)
                        send_to_airtable(record_id, {"error": error_message})
                else:
                    error_message = "Extraction failed. Please check the PDF file."
                    print(error_message)
                    send_to_airtable(record_id, {"error": error_message})

        except Exception as e:
            error_message = f"An error occurred during processing: {str(e)}"
            print(error_message)
            send_to_airtable(record_id, {"error": error_message})
            raise  # Re-raise the exception for logging

    thread = Thread(target=process)
    thread.start()

@app.route('/process_pdf', methods=['POST'])
def process_pdf_route():
    data = request.json  # Assuming Airtable sends JSON data
    pdf_url = data.get('pdf_url')
    record_id = data.get('record_id')
    custom_prompt = data.get('custom_prompt')
    response_schema = data.get('response_schema')

    if pdf_url and record_id and response_schema:
        try:
            process_pdf_async(pdf_url, record_id, custom_prompt, response_schema)
            return jsonify({"status": "processing started"}), 200
        except json.JSONDecodeError:
            return jsonify({"error": "Invalid JSON format in response_schema"}), 400
    else:
        return jsonify({"error": "Missing pdf_url, record_id, or response_schema"}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)

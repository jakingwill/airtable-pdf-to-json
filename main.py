import os
import google.generativeai as genai
import requests
from flask import Flask, request, jsonify
from threading import Thread
import uuid
import fitz  # PyMuPDF
import pathlib
import tqdm

app = Flask(__name__)

# Initialize Gemini API client with your API key
gemini_api_key = os.getenv('GEMINI_API_KEY')
client = genai.configure(api_key=gemini_api_key)
airtable_webhook_url = os.environ['AIRTABLE_WEBHOOK']

def extract_pdf_content(pdf_path, output_dir):
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_document = fitz.open(pdf_path)
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap()
        image_filename = output_dir / f"image-{page_num + 1}.jpg"
        pix.save(image_filename)
        print(f"Saved {image_filename}")
    return [str(f) for f in output_dir.glob('*.jpg')]

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
        print(f"Uploaded {img} as {file.uri}")
    return files

def summarize_content(files, custom_prompt, response_schema):
    prompt = [custom_prompt] + [file.uri for file in files] + ["[END]\n\nPlease extract the text from these images."]
    model = genai.GenerativeModel(model_name='models/gemini-1.5-flash')
    
    # Use GenerationConfig to get JSON response with the desired schema
    generation_config = genai.GenerationConfig(
        response_mime_type='application/json',
        response_schema=response_schema
    )
    
    response = model.generate_content(prompt, generation_config=generation_config)
    # Access the 'question' attribute from the response correctly
    return response

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
            unique_id = str(uuid.uuid4())
            request_dir = pathlib.Path('requests') / unique_id
            request_dir.mkdir(parents=True, exist_ok=True)

            pdf_path = download_pdf(pdf_url, request_dir)
            output_dir = request_dir / 'output'
            image_files = extract_pdf_content(pdf_path, output_dir)

            if image_files:
                files = upload_to_gemini(image_files)
                if files:
                    summary = summarize_content(files, custom_prompt, response_schema)
                    print(f"Summary for record_id {record_id}:\n{summary}")
                    send_to_airtable(record_id, summary)
                else:
                    print("No files uploaded to Gemini.")
            else:
                print("Extraction failed. Please check the PDF file.")
        except Exception as e:
            print(f"An error occurred during processing: {e}")

    thread = Thread(target=process)
    thread.start()

@app.route('/process_pdf', methods=['POST'])
def process_pdf_route():
    data = request.get_json()
    pdf_url = data.get('pdf_url')
    record_id = data.get('record_id')
    custom_prompt = data.get('custom_prompt')
    response_schema = data.get('response_schema')

    if pdf_url and record_id and response_schema:
        process_pdf_async(pdf_url, record_id, custom_prompt, response_schema)
        return jsonify({"status": "processing started"}), 200
    else:
        return jsonify({"error": "Missing pdf_url, record_id, or response_schema"}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)

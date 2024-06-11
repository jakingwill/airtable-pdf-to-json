import os
import subprocess
import google.generativeai as genai
import requests
from flask import Flask, request, jsonify
from threading import Thread
import uuid

app = Flask(__name__)

# Initialize Gemini API client with your API key
gemini_api_key = os.environ['GEMINI_API_KEY']
client = genai.configure(api_key=gemini_api_key)
airtable_webhook_url = os.environ['AIRTABLE_WEBHOOK']

# Define functions to upload text and image files to Gemini
def upload_text_to_gemini(text_content):
    with open('temp_text.txt', 'w') as file:
        file.write(text_content)
    response = genai.upload_file('temp_text.txt')
    os.remove('temp_text.txt')
    return response.uri

def upload_image_to_gemini(image_file_path):
    with open(image_file_path, 'rb') as file:
        response = genai.upload_file(image_file_path)
    return response.uri

# Define function to extract text and images from PDF
def extract_pdf_content(pdf_path, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Extract images using pdftoppm for the full PDF
    subprocess.run(['pdftoppm', '-jpeg', pdf_path, os.path.join(output_dir, 'images')])

    # Extract text using pdftotext for the full PDF
    subprocess.run(['pdftotext', pdf_path, os.path.join(output_dir, 'text.txt')])

    # Return the path of the extracted text file
    return os.path.join(output_dir, 'text.txt')

def download_pdf(pdf_url, download_folder):
    response = requests.get(pdf_url)
    if response.status_code == 200:
        file_path = os.path.join(download_folder, 'downloaded_pdf.pdf')
        with open(file_path, 'wb') as file:
            file.write(response.content)
        return file_path
    else:
        raise Exception(f"Failed to download PDF, status code: {response.status_code}")

# Define function to upload extracted content to Gemini
def upload_to_gemini(output_dir):
    files = []

    # Print list of files in output directory for debugging
    print("Files in output directory:")
    print(os.listdir(output_dir))

    # Upload image files to Gemini
    for filename in sorted(os.listdir(output_dir)):
        if filename.startswith('images-'):
            image_file_path = os.path.join(output_dir, filename)
            image_gemini_file = upload_image_to_gemini(image_file_path)
            files.append(image_gemini_file)

    # Upload text file to Gemini
    text_file_path = os.path.join(output_dir, 'text.txt')
    if os.path.exists(text_file_path):
        with open(text_file_path, 'r') as file:
            text_content = file.read()
            text_gemini_file = upload_text_to_gemini(text_content)
            files.append(text_gemini_file)
    else:
        print("Error: Text file not found.")

    return files

# Define function to summarize content using Gemini
def summarize_content(files, custom_prompt):
    # Prepare prompt with text and image references
    prompt = [custom_prompt]
    prompt.extend(files)
    prompt.append("[END]\n\nHere is the document")

    # Generate content using Gemini
    model = genai.GenerativeModel(model_name='gemini-1.5-flash')
    response = model.generate_content(prompt)

    return response.text

def send_to_airtable(record_id, summary):
    webhook_url = airtable_webhook_url
    data = {
        "record_id": record_id,
        "summary": summary
    }
    response = requests.post(webhook_url, json(data))
    if response.status_code == 200:
        print("Successfully sent data to Airtable")
    else:
        print(f"Failed to send data to Airtable: {response.status_code}, {response.text}")

# Function to process the PDF asynchronously
def process_pdf_async(pdf_url, record_id, custom_prompt):
    def process():
        try:
            print(f"Received pdf_url: {pdf_url}")
            print(f"Received record_id: {record_id}")
            print(f"Received custom_prompt: {custom_prompt}")

            # Create unique directory for each request
            unique_id = str(uuid.uuid4())
            request_dir = os.path.join('requests', unique_id)
            os.makedirs(request_dir, exist_ok=True)

            # Define the path to save the downloaded PDF
            pdf_path = os.path.join(request_dir, 'downloaded_pdf.pdf')
            print(pdf_path)

            # Download PDF from the provided URL
            pdf_path = download_pdf(pdf_url, request_dir)

            # Extract text from the downloaded PDF
            output_dir = os.path.join(request_dir, 'output')
            text_file_path = extract_pdf_content(pdf_path, output_dir)

            if text_file_path:
                # Upload extracted content to Gemini
                files = upload_to_gemini(output_dir)

                # Read the assessment text directly from the text file
                assessment_text_path = os.path.join(output_dir, 'text.txt')
                with open(assessment_text_path, 'r') as file:
                    assessment_text = file.read()

                # Summarize content using Gemini with custom prompt
                summary = summarize_content([assessment_text], custom_prompt)
                print(summary)

                # Send summary to Airtable
                send_to_airtable(record_id, summary)
            else:
                print("Extraction failed. Please check the PDF file.")
        except Exception as e:
            print(f"An error occurred during processing: {e}")

    # Start a new thread to process the PDF
    thread = Thread(target=process)
    thread.start()

@app.route('/process_pdf', methods=['POST'])
def process_pdf_route():
    data = request.get_json()
    print(f"Received data: {data}")
    pdf_url = data.get('pdf_url')
    record_id = data.get('record_id')
    custom_prompt = data.get('custom_prompt')

    if not pdf_url:
        print("Missing pdf_url")
    if not record_id:
        print("Missing record_id")

    if pdf_url and record_id:
        process_pdf_async(pdf_url, record_id, custom_prompt)
        return jsonify({"status": "processing started"}), 200
    else:
        return jsonify({"error": "Missing pdf_url or record_id"}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)

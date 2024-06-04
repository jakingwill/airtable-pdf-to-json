import os
import subprocess
import shutil
import google.generativeai as genai
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

# Initialize Gemini API client with your API key
gemini_api_key = os.environ['GEMINI_API_KEY']
client = genai.configure(api_key=gemini_api_key)
airtable_webhook_url = os.environ['AIRTABLE_WEBHOOK']

# Simplified process_pdf function that only logs the received data
def process_pdf(pdf_url, record_id):
    print(f"Received pdf_url: {pdf_url}")
    print(f"Received record_id: {record_id}")

@app.route('/process_pdf', methods=['POST'])
def process_pdf_route():
    data = request.get_json()
    print(f"Received data: {data}")  # Log the received data for debugging
    pdf_url = data.get('pdf_url')
    record_id = data.get('record_id')

    if not pdf_url:
        print("Missing pdf_url")
    if not record_id:
        print("Missing record_id")

    if pdf_url and record_id:
        process_pdf(pdf_url, record_id)
        return jsonify({"status": "processing started"}), 200
    else:
        return jsonify({"error": "Missing pdf_url or record_id"}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)


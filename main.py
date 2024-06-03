import os
import subprocess
import shutil
import google.generativeai as genai
import requests

# Initialize Gemini API client with your API key
gemini_api_key = os.environ['GEMINI_API_KEY']
client = genai.configure(api_key=gemini_api_key)

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


def download_pdf(pdf_url, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Extract the filename from the URL
    filename = 'downloaded_pdf.pdf'

    # Download the PDF file
    file_path = os.path.join(output_dir, filename)
    with open(file_path, 'wb') as f:
        response = requests.get(pdf_url)
        f.write(response.content)

    return file_path

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
def summarize_content(files):
    # Prepare prompt with text and image references
    prompt = ['here is the assessment file']
    prompt.extend(files)
    prompt.append("[END]\n\nYou are a document entity extraction specialist for a school that gives you assessments. Given an assessment, your task is to extract the text value of the following entities:\n{\n \"question\": [\n  {\n   \"question_number\": \"\",\n   \"total_marks\": \"\",\n   \"question_text\": \"\",\n   \"marking_guide\": \"\"\n  }\n ],\n \"answer\": [\n  {\n   \"question_number\": \"\",\n   \"student_answer\": \"\"\n  }\n ],\n}\n\n- The JSON schema must be followed during the extraction.\n- The values must only include text strings found in the document.\n- Generate null for missing entities.")

    # Generate content using Gemini
    model = genai.GenerativeModel(model_name='gemini-1.5-flash')
    response = model.generate_content(prompt)

    return response.text


def main(pdf_url):
    # Create a 'downloads' directory if it doesn't exist
    os.makedirs('downloads', exist_ok=True)

    # Define the path to save the downloaded PDF
    pdf_path = os.path.join('downloads', 'downloaded_pdf.pdf')
    print(pdf_path)

    # Download PDF from the provided URL
    pdf_path = download_pdf(pdf_url, 'downloads')

    # Extract text from the downloaded PDF
    output_dir = 'output'
    text_file_path = extract_pdf_content(pdf_path, output_dir)

    if text_file_path:
        # Upload extracted content to Gemini
        files = upload_to_gemini(output_dir)

        # Read the assessment text directly from the text file
        assessment_text_path = os.path.join(output_dir, 'text.txt')
        with open(assessment_text_path, 'r') as file:
            assessment_text = file.read()

        # Summarize content using Gemini
        summary = summarize_content([assessment_text])
        print(summary)
    else:
        print("Extraction failed. Please check the PDF file.")

if __name__ == '__main__':
    pdf_url = 'https://www.cardiffmet.ac.uk/education/courses/Documents/+Numeracy%20test%20%20A2%20EXEMPLAR%20SEWCTET%20June%202013.pdf'  # Replace with the actual URL from Airtable
    main(pdf_url)

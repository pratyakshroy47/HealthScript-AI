import os
import json
import io
import uuid
import traceback
from PIL import Image
from google.cloud import documentai_v1beta3 as documentai
from google.cloud.documentai_v1beta3 import DocumentProcessorServiceClient
from google.api_core.client_options import ClientOptions
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from google.oauth2 import service_account

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration variables
project_id = "gpu-reservation-sarvam"
location = "eu"
processor_id = "3b7bc11cd5be5a6a"
processor_version_id = "pretrained-ocr-v2.0-2023-06-02"
service_account_path = "/home/ext_pratyakshroy47_gmail_com/sanskrit-text-extractor/gpu-reservation-sarvam-bdb5c9cf6486.json"
output_folder = "/home/ext_pratyakshroy47_gmail_com/medical-fhir/medical_pipeline/ocr-of-prescription/prescription_jsons"
image_folder_path = "/home/ext_pratyakshroy47_gmail_com/medical-fhir/medical_pipeline/image_preprocessing/postproc_images"

# Set up credentials programmatically
credentials = service_account.Credentials.from_service_account_file(service_account_path)

# Set endpoint to EU
client_options = ClientOptions(api_endpoint="eu-documentai.googleapis.com:443")

def get_processed_files(output_folder):
    """Get a set of base filenames for files that have already been processed."""
    return {file.split('.')[0] for file in os.listdir(output_folder) if file.endswith('.json')}


def layout_to_text(layout, text):
    logging.info("Converting layout to text.")
    return "".join(
        text[text_segment.start_index:text_segment.end_index]
        for text_segment in layout.text_anchor.text_segments
    )

def detect_text_document_ai(image_path):
    logging.info(f"Processing image for text detection: {image_path}")
    client = DocumentProcessorServiceClient(credentials=credentials, client_options=client_options)
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()

    input_config = documentai.types.RawDocument(content=content, mime_type='image/png')
    request = documentai.types.ProcessRequest(
        name=f'projects/{project_id}/locations/{location}/processors/{processor_id}/processorVersions/{processor_version_id}',
        raw_document=input_config
    )

    try:
        result = client.process_document(request=request)
        logging.info("Document processing successful.")
        return [layout_to_text(paragraph.layout, result.document.text) for page in result.document.pages for paragraph in page.paragraphs]
    except Exception as e:
        logging.error(f"Failed to process document: {e}")
        return []
    

Image.MAX_IMAGE_PIXELS = None

def save_to_json(image_path, text_blocks):
    logging.info("Saving extracted text to JSON file.")
    os.makedirs(output_folder, exist_ok=True)
    base_filename = os.path.basename(image_path)
    filename_without_extension = os.path.splitext(base_filename)[0]
    output_file = os.path.join(output_folder, f"{filename_without_extension}.json")
    data = {
        'source': image_path,
        'text': text_blocks,
        'word_count': sum(len(block.split()) for block in text_blocks)
    }
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    logging.info(f"Data saved to {output_file}")

def process_image(image_path):
    logging.info(f"Starting processing of image: {image_path}")
    text_blocks = detect_text_document_ai(image_path)
    if text_blocks:
        save_to_json(image_path, text_blocks)
        logging.info(f"Successfully processed and saved data from {image_path}")

if __name__ == "__main__":
    processed_files = get_processed_files(output_folder)
    image_files = [f for f in os.listdir(image_folder_path) if f.endswith('.png') and f.split('.')[0] not in processed_files]
    
    image_paths = [os.path.join(image_folder_path, image_file) for image_file in image_files]
    with ThreadPoolExecutor(max_workers=95) as executor:
        future_to_image = {executor.submit(process_image, image_path): image_path for image_path in image_paths}
        for future in as_completed(future_to_image):
            image_path = future_to_image[future]
            logging.info(f"Finished processing {image_path}")

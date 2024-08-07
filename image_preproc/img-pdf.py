import os
from PIL import Image
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def compress_image(image_path, max_size_bytes, output_folder):
    logging.info(f"Compressing {image_path} to fit within {max_size_bytes} bytes.")
    try:
        img = Image.open(image_path)
        base_filename = os.path.basename(image_path)
        filename_without_extension = os.path.splitext(base_filename)[0]
        output_file = os.path.join(output_folder, f"{filename_without_extension}.compressed.png")

        # Reduce the quality until the size is below the max_size_bytes
        quality = 95
        while quality > 10:
            img.save(output_file, format="PNG", quality=quality, optimize=True)
            if os.path.getsize(output_file) < max_size_bytes:
                break
            quality -= 5

        if os.path.getsize(output_file) >= max_size_bytes:
            logging.warning(f"Could not compress {image_path} to below {max_size_bytes} bytes.")
            return None
        logging.info(f"Successfully compressed {image_path} to {output_file}.")
        return output_file
    except Exception as e:
        logging.error(f"Failed to compress {image_path}: {e}")
        return None

def convert_image_to_pdf(image_path, output_folder):
    logging.info(f"Converting {image_path} to PDF.")
    try:
        image = Image.open(image_path)
        rgb_image = image.convert('RGB')  # Convert image to RGB
        base_filename = os.path.basename(image_path)
        filename_without_extension = os.path.splitext(base_filename)[0]
        output_file = os.path.join(output_folder, f"{filename_without_extension}.pdf")
        rgb_image.save(output_file, "PDF", quality=95)
        logging.info(f"Successfully converted {image_path} to {output_file}.")
    except Exception as e:
        logging.error(f"Failed to convert {image_path} to PDF: {e}")

def convert_images_in_folder(input_folder, output_folder, max_size_bytes):
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']
    image_files = [f for f in os.listdir(input_folder) if os.path.splitext(f)[1].lower() in image_extensions]

    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        if os.path.getsize(image_path) > max_size_bytes:
            logging.info(f"File {image_path} exceeds {max_size_bytes} bytes and will be compressed.")
            compressed_image_path = compress_image(image_path, max_size_bytes, output_folder)
            if compressed_image_path:
                convert_image_to_pdf(compressed_image_path, output_folder)
        else:
            convert_image_to_pdf(image_path, output_folder)

if __name__ == "__main__":
    input_folder = '/home/ext_pratyakshroy47_gmail_com/medical-fhir/medical_pipeline/image_preprocessing/postproc_images'  # Change this to your input folder
    output_folder = '/home/ext_pratyakshroy47_gmail_com/medical-fhir/medical_pipeline/image_preprocessing/prescription_pdfs'  # Change this to your output folder
    max_size_bytes = 20 * 1024 * 1024  # Maximum file size in bytes (e.g., 20 MB)

    convert_images_in_folder(input_folder, output_folder, max_size_bytes)

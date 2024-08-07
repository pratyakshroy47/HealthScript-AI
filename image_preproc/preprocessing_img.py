import cv2
import numpy as np
import os
from PIL import Image
import tempfile
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def check_and_convert_to_rgb(image):
    """Check and convert image to RGB if it's in BGR format."""
    if image.shape[-1] == 3:  # Image has 3 channels
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def convert_to_grayscale(image):
    """Convert an RGB or BGR image to grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def apply_median_blur(image):
    """Apply median blur to reduce noise while preserving edges."""
    return cv2.medianBlur(image, 5)  # Adjusted kernel size for better noise reduction

def normalize_image(image):
    """Normalize the image to the range 0-255."""
    return cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

def equalize_histogram(image):
    """Apply histogram equalization to enhance image contrast."""
    return cv2.equalizeHist(image)

def remove_noise(image):
    """Remove noise from the image using Non-Local Means Denoising."""
    return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)  # Adjusted parameters for better denoising

def thinning_and_skeletonization(image):
    """Apply thinning and skeletonization to make stroke widths uniform."""
    kernel = np.ones((1, 1), np.uint8)  # Smaller kernel for more subtle thinning
    return cv2.erode(image, kernel, iterations=2)

def set_image_dpi(file_path):
    """Set the image DPI to 300 if it is not already."""
    with Image.open(file_path) as im:
        length_x, width_y = im.size
        factor = max(1, float(300.0 / (im.info.get('dpi', (72, 72))[0])))  # Default DPI is 72 if not set
        size = int(factor * length_x), int(factor * width_y)
        im_resized = im.resize(size, Image.Resampling.LANCZOS)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        temp_filename = temp_file.name
        im_resized.save(temp_filename, dpi=(300, 300))
    return temp_filename

def preprocess_image(image_path, output_folder):
    """Preprocess the image for better OCR performance."""
    logging.info(f"Starting preprocessing for {image_path}")
    temp_filename = set_image_dpi(image_path)
    image = cv2.imread(temp_filename)
    noise_removed_image = remove_noise(image)
    rgb_image = check_and_convert_to_rgb(noise_removed_image)
    gray_image = convert_to_grayscale(rgb_image)
    blurred_image = apply_median_blur(gray_image)
    normalized_image = normalize_image(blurred_image)
    equalized_image = equalize_histogram(normalized_image)
    thinned_image = thinning_and_skeletonization(equalized_image)

    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save the processed image to the specified folder
    output_file_path = os.path.join(output_folder, os.path.basename(image_path))
    cv2.imwrite(output_file_path, thinned_image)
    os.unlink(temp_filename)  # Remove the temporary file
    logging.info(f"Finished preprocessing {image_path}, saved to {output_file_path}")
    return output_file_path

def batch_process_images(input_folder, output_folder):
    """Process all images in the input folder and save the outputs to the output folder."""
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            image_path = os.path.join(input_folder, filename)
            output_path = preprocess_image(image_path, output_folder)
            logging.info(f"Processed {filename} and saved to {output_path}")

# Example usage
input_folder_path = '/home/ext_pratyakshroy47_gmail_com/medical-fhir/medical_pipeline/prescription_images'
output_folder_path = '/home/ext_pratyakshroy47_gmail_com/medical-fhir/medical_pipeline/image_preprocessing/postproc_images'
batch_process_images(input_folder_path, output_folder_path)

# HealthScript AI

## Project Description

The **HealthScript AI** is an innovative solution for digitizing handwritten medical prescriptions commonly used in Indian hospitals. This pipeline integrates Optical Character Recognition (OCR), speech-to-text transcription, and machine learning models to convert unstructured medical data into a structured FHIR-compliant digital format. The pipeline enhances the accuracy and accessibility of medical information, supporting healthcare providers, pharmacies, and patients.

## Features

- **Image Processing**: High-quality text extraction from prescription images.
- **Audio Transcription**: Transcribe doctor-patient conversations with Whisper V3 for contextual understanding.
- **Data Cross-Referencing**: Compare image, OCR text, and audio data for accurate extraction.
- **FHIR Compliance**: Generate structured digital prescriptions in FHIR format.
- **Error Handling**: Identify and highlight missing or ambiguous information.
- **Human Thought Process Integration**: Mimic data science engineers' approach for validation and decision-making.

## Architecture

The pipeline is composed of several stages:

1. **Image Preprocessing**: Enhancing image quality using OpenCV for better OCR accuracy.
2. **OCR Extraction**: Utilizing Document AI for precise text extraction from images.
3. **Audio Transcription**: Converting audio recordings to text using Whisper V3.
4. **Cross-Referencing**: Comparing extracted data across image, OCR, and audio inputs.
5. **VLM/LLM Processing**: Employing GPT-4o and Claude Opus for intelligent analysis and digitization.
6. **FHIR Generation**: Structuring the output in FHIR format for integration with healthcare systems.


## Installation

To set up the Medical FHIR Pipeline locally, follow these steps:

### Prerequisites

- **Python 3.11+**
- **Git**
- **Virtual Environment**

### Clone the Repository
 ```bash
 git clone https://github.com/your-username/medical-fhir-pipeline.git
 cd medical-fhir-pipeline

 #Set Up a Virtual Environment
 git clone https://github.com/your-username/medical-fhir-pipeline.git
 cd medical-fhir-pipeline

 # Install Dependencies
 pip install -r requirements.txt
```

## Output Structure
``` bash
{
  "Patient": {
    "Name": "Rajesh Kumar",
    "Age": "45",
    "Symptoms": "Fever, Cough, Headache"
  },
  "Diagnosis": "Viral Fever",
  "Medicines": [
    {
      "Name": "Amoxicillin 500mg",
      "Dosage": "Three times a day",
      "Instructions": "After meals"
    },
    {
      "Name": "Paracetamol 500mg",
      "Dosage": "Once a day",
      "Instructions": "Before bedtime"
    }
  ],
  "Doctor": "Dr. A. Sharma"
}
```

## Technologies Used

Python: Programming language for core logic.
OpenCV: Image processing library.
Document AI: OCR tool for text extraction.
Whisper V3: Audio-to-text transcription model.
GPT-4o and Claude Opus: Advanced language models for analysis.
FHIR: Standard for healthcare information exchange.

import json
import os
import logging
import base64
from openai import OpenAI
import anthropic

# Initialize API clients
openai_client = OpenAI(api_key="SECRET_KEY_OPENAI")
claude_client = anthropic.Anthropic(api_key="SECRET_KEY_CLAUDE")


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def encode_image(image_path):
    """Encode image to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def read_ocr_text(ocr_path):
    """Read OCR text from a JSON file."""
    with open(ocr_path, 'r') as file:
        data = json.load(file)
    return data.get('text', '')  # Assuming 'text' is the key containing OCR data

def generate_output_from_apis(ocr_text, audio_transcription, base64_image, prompt, image_path, ocr_path, audio_transcription_path):
    logging.info("Sending data to both GPT and Claude for combined outputs...")
    logging.debug(f"Files being processed: Image: {image_path}, OCR: {ocr_path}, Audio: {audio_transcription_path}")
    
    # Ensure OCR text is correctly formatted as a string
    ocr_text = str(ocr_text)  # Convert OCR text to string if not already
    
    # Correct message format for GPT including OCR text and audio transcription
    messages_gpt = [
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
        {"role": "user", "content": [{"type": "text", "text": ocr_text}]},  # OCR text
        {"role": "user", "content": [{"type": "text", "text": audio_transcription}]},  # Audio transcription
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]}
    ]

    # Correct message format for Claude including OCR text and audio transcription
    messages_claude = [
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Please analyze the image, OCR and audio transcription."}]},
        {"role": "user", "content": [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64_image}}]},
        {"role": "assistant", "content": [{"type": "text", "text": ocr_text}]},  # OCR text
        {"role": "user", "content": [{"type": "text", "text": audio_transcription}]}  # Audio transcription
    ]

    # Call GPT and Claude with updated messages
    try:
        logging.info("Sending request to GPT API...")
        response_gpt = openai_client.chat.completions.create(
            model="gpt-4o", messages=messages_gpt, temperature=0.0, max_tokens=4096)
        output_gpt = response_gpt.choices[0].message.content
        logging.info("Received output from GPT")
    except Exception as e:
        logging.error(f"GPT request failed: {e}")
        output_gpt = "GPT request failed"

    try:
        logging.info("Sending request to Claude API...")
        response_claude = claude_client.messages.create(
            model="claude-3-opus-20240229", messages=messages_claude, max_tokens=4096, temperature=0.0)
        output_claude = "\n".join([content_block.text for content_block in response_claude.content])
        logging.info("Received output from Claude")
    except Exception as e:
        logging.error(f"Claude request failed: {e}")
        output_claude = "Claude request failed"

    return output_gpt, output_claude

def save_output_to_files(gpt_output, claude_output, base_filename):
    """Save outputs to text files."""
    gpt_file_path = f"{base_filename}_gpt_output.txt"
    claude_file_path = f"{base_filename}_claude_output.txt"

    # Save GPT output
    with open(gpt_file_path, "w") as file:
        file.write(gpt_output)
    logging.info(f"GPT output saved to {gpt_file_path}")

    # Save Claude output
    with open(claude_file_path, "w") as file:
        file.write(claude_output)
    logging.info(f"Claude output saved to {claude_file_path}")

# Main function adjustments
def main(image_path, ocr_path, audio_transcription_path):
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    base64_image = encode_image(image_path)
    ocr_text = read_ocr_text(ocr_path)  # Read OCR text from JSON
    # Load audio transcription (assuming it's stored as plain text for simplicity)
    with open(audio_transcription_path, 'r') as file:
        audio_transcription = file.read()

    prompt = f"""
**Context:**
I am developing a prescription transcriber that interprets and extracts information from handwritten prescriptions commonly used in Indian hospitals.
The input will be an image of the prescription, and the output should have all the data that could be extracted from the image.

Following can be used to improve accuracy if the image is unclear, but the prescription image is the primary source of truth. And I am also providing OCR text extracted from the image of the prescription.

**Task:**
- Analyze the provided image of the handwritten prescription.
- Analyze the OCR text extracted from the prescription of the image.
- Analyze the transcript of the conversation between the doctor and patient
- Using all 3 input sources, extract all identifiable information sources such as patient name, age, date, symptoms reported (C/O: Complain of), history (H/O: History of), allergy, investigation or examination findings (Ix or O/E), diagnosis (Dx: Diagnosis), medical tests, prescribed medicines, dosages, and doctor's name.
- Clearly state any assumptions made to identify or interpret the information more accurately.
- If any information cannot be identified or retrieved, highlight this with the reason why it could not be identified.
- Cross-reference your output with the extracted text OCR, transcript of the patient-doctor conversation and then take a judgement on final extracted data. Also state your judgement or reasoning for the extracted value very clearly.

**Thought Process / Analysis**
- You have three parameters: the prescription image, OCR text of the prescription, and the audio transcript of the doctor and patient conversation.
- If you give all these data to a data science engineer and ask them to provide the desired output (correct digitization of medical prescriptions), their thought process would be as follows:
- Comparison and Validation: Compare all three inputs to identify the closest or correct information. For instance, if the OCR text reads a medicine name as "paractamel," the prescription image as "paractmol," and the audio transcript as "paracetamol," the engineer would identify "paracetamol" as the correct spelling. After comparing all three inputs, it is important to also verify the medicine names or any information using its own training data. For instance, if there are spelling mistakes in a particular word across all three inputs, the system should compare it with its training data to find the correct spelling and confirm if it matches the symptoms or diagnosis. It should also check if the name is similar to any wrong names present in the inputs. Other factors should also be considered to determine the accurate medicine name or data originally present in the prescription.
- Cross-Referencing: The accurate word or sentence can be present in any parameter, so all data must be compared before generating the analysis tokens. This comparison ensures that the most accurate information is processed and used to generate the final output.
- Assumptions and Reasoning: Clearly state any assumptions made to identify or interpret the information more accurately.
- Error Handling: If any information cannot be identified or retrieved, highlight this with the reason why it could not be identified.

**Output:**
- Provide a structured output with headings for each type of information.
- Include any assumptions made during the extraction process.
- Clearly state any information that could not be identified and the reason for it.
- Reference OCR parsing results to augment the accuracy of the extracted information if needed.

**Example:**
- Patient Name: [Extracted Name] (Assumption: Based on the placement at the top of the prescription)
- Age: [Extracted Age] (Assumption: Age is typically written next to the name)
- Date: [Extracted Date]
- C/O (Complain of): [Extracted Symptoms]
- H/O (History of): [Extracted History]
- Allergy: [Extracted Allergy]
- Ix (Investigation) / O/E (On Examination): [Extracted Investigation or Examination Findings]
- Dx (Diagnosis): [Extracted Diagnosis]
- Medical Test: [Extracted Tests and Procedures and results if found in the prescription image or OCR text, or audio transcript ]
- Medicines / Advised / Adv / Rx:
    - Medicine 1: [Name] - DosageInstruction: [Dosage with primary instruction] - DosageAdditionalInstruction : [Dosage with additional instruction] - Dosage Method : [Dosage method and route] (Assumption: Dosage interpreted based on common medical abbreviations)
    - Medicine 2: [Name] - DosageInstruction: [Dosage with primary instruction] - DosageAdditionalInstruction : [Dosage with additional instruction] - Dosage Method : [Dosage method and route]
    - Doctor's Name: [Extracted Name] (Assumption: Name identified based on typical signature placement)

**Dosage Interpretation:**
Identify dosages by examining the number of circles made by the doctor when no numbers are present. For example, 3 circles mean thrice a day. Numbers signify dosage, and circles represent no dosage.
Additional abbreviations commonly used by doctors in medical prescriptions:
- BF / BBF: Take medicine "before breakfast"
- HS: Take medicine "1 hour before sleeping"
- STAT: Urgent
- SOS: si opus sit meaning "if necessary"
- TDS: Ter Die Sumendum meaning "3 times a day"
- Adv: Advised to
- Rx : prescribed to
- Dx: Diagnosis
- H/O or Hx: History of
- Tab: Tablets
- Cap: Capsules
- Syp: Syrup
- Rx: it denotes to the pharmacist or take
- Ac: ante cibum means before meal.
- Pc: post cibum means after meal.
- Hs: hora somni means at bedtime or night.
- BID/BD: twice daily.
- TID/TD: thrice daily.
- QID: 4 times per day.
- EOD: every other day. Example- 1tab BD PC that means 1tablet twice daily after meal patient have to take.
- SC / SQ: subcutaneous.
- ID: Intradermal.
- IM: intramuscular.
- IN: intranasal.
- qw / q.wk: weekly
- ud: as directed
- qhs: every night at bedtime
- q: every
- qod: every other day
- qh: every hour
- TW: twice a week
- IV: intravenous
- Ex: out of
- OD: Once a day
- PRN: as needed
- tsp: teaspoon (5ml)
- tbsp: tablespoon (15ml)
- OH: Orthopedically Handicapped
- G2P1: two pregancies, one birth
- VAS: pain rating scale
- PO: by mouth
- SOB: shortness of breath
- p/a: per abdomen
- SpOz: estimate of arterial oxygen saturation

**Unidentified Information:**
- Patient Address: Unable to identify due to illegible handwriting
- Dosage: Unable to identify due to Indic language input or written in Bengali
[Other examples of unidentified information with reasons]

**Example Image Analysis:**
- Patient Name: Rajesh Kumar (Assumption: Name identified based on its position at the top)
- Age: 45 (Assumption: Commonly written next to the name)
- Date: 03/06/2024
- C/O (Complain of): "Fever, Cough, Headache"
- H/O (History of): "Diabetes, Hypertension"
- Allergy: "Penicillin"
- Ix (Investigation) / O/E (On Examination): "Blood Test - Malaria -ve"
- Dx (Diagnosis): "Viral Fever" [include the extracted diagnosis here]
- Doctor's Comments / Suggestions:
    - Take neurology consultation
    - control Glycemic index (GI)
- Medicines / Advised / Adv / Rx:
    - Medicine 1: "Amoxicillin 500mg"- DosageInstruction: "three times a day" - DosageAdditionalInstruction : "after dinner" - Dosage Method : "be taken orally" (Assumption: Dosage derived from standard prescription format) (Confirmed by OCR text and audio transcript)
    - Medicine 2: "Paracetamol 500mg" - DosageInstruction: "once a day for 4 weeks" - DosageAdditionalInstruction : "before lunch and dinner" - Dosage Method : "orally" (Assumption: Dosage frequency inferred from standard abbreviations) (Confirmed by Prescription image and audio transcript)
    - Medicine 3: "Aspirin 325mg" - DosageInstruction: "twice a day, if necessary" - DosageAdditionalInstruction : "before bedtime" - Dosage Method : "orally" (Assumption: Dosage frequency inferred from standard abbreviations) (Confirmed by Prescription image)
    - Doctor's Name: Dr. A. Sharma (Assumption: Name identified based on signature placement)
- Unidentified Information:
    - Patient Address: Unable to identify due to illegible handwriting
    - Dosage: Unable to identify due to Indic language input or written in Bengali

Note:
The OCR results provided may contain mistakes but serve as a supportive tool to enhance the accuracy of the transcription from the image. The prescription image is the primary source of truth, but in cases of uncertainty, the OCR results and audio transcription can be used to refine the answer. At the end of the output, please detail which inputs were processed to produce the output and which were not. Also, specify how each input contributed to the final result, and mention this for every output.
"""

    # Generate and save outputs
    gpt_output, claude_output = generate_output_from_apis(ocr_text, audio_transcription, base64_image, prompt, image_path, ocr_path, audio_transcription_path)
    save_output_to_files(gpt_output, claude_output, base_filename)
    
if __name__ == "__main__":
    image_folder = "/home/ext_pratyakshroy47_gmail_com/medical-fhir/medical_pipeline/imgsssss/postproc_images"
    ocr_folder = "/home/ext_pratyakshroy47_gmail_com/medical-fhir/medical_pipeline/ocr_jsonsssss"
    audio_folder = "/home/ext_pratyakshroy47_gmail_com/medical-fhir/medical_pipeline/audiooooooooo"

    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    ocr_files = [f for f in os.listdir(ocr_folder) if f.endswith('.json')]
    audio_files = [f for f in os.listdir(audio_folder) if f.endswith('.txt')]

    # Ensure all files are sorted to match by prescription
    image_files.sort()
    ocr_files.sort()
    audio_files.sort()

    for image_file, ocr_file, audio_file in zip(image_files, ocr_files, audio_files):
        image_path = os.path.join(image_folder, image_file)
        ocr_path = os.path.join(ocr_folder, ocr_file)
        audio_transcription_path = os.path.join(audio_folder, audio_file)

        logging.info(f"Processing files - Image: {image_path}, OCR: {ocr_path}, Audio: {audio_transcription_path}")
        main(image_path, ocr_path, audio_transcription_path)


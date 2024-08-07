import os
from pathlib import Path
from pydub import AudioSegment
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def convert_audio_to_wav(input_path, output_path):
    """
    Convert audio file to .wav format and save to the specified output path.
    """
    try:
        audio = AudioSegment.from_file(input_path)
        audio.export(output_path, format='wav')
        logging.info(f"Converted {input_path} to {output_path}")
        return True
    except Exception as e:
        logging.error(f"Failed to convert {input_path}: {e}")
        return False

def convert_audio_files(input_folder, output_folder):
    """
    Convert all audio files in the input folder to .wav format and save them to the output folder.
    """
    input_folder_path = Path(input_folder)
    output_folder_path = Path(output_folder)
    output_folder_path.mkdir(parents=True, exist_ok=True)

    supported_formats = ('.mp3', '.flac', '.ogg', '.m4a', '.aac', '.wma', '.wav')

    audio_files = [audio_file for audio_file in input_folder_path.rglob('*') if audio_file.suffix.lower() in supported_formats]

    logging.info(f"Found {len(audio_files)} audio files to convert.")

    with ThreadPoolExecutor() as executor:
        future_to_audio = {executor.submit(convert_audio_to_wav, audio_file, output_folder_path / (audio_file.stem + '.wav')): audio_file for audio_file in audio_files}

        for future in tqdm(as_completed(future_to_audio), total=len(future_to_audio), desc="Converting files", unit="file"):
            audio_file = future_to_audio[future]
            try:
                future.result()
            except Exception as e:
                logging.error(f"Error processing {audio_file}: {e}")

if __name__ == "__main__":
    input_folder = '/home/ext_pratyakshroy47_gmail_com/medical-fhir/medical_pipeline/audio_transcripter/junk-audio'
    output_folder = '/home/ext_pratyakshroy47_gmail_com/medical-fhir/medical_pipeline/audio_transcripter/transcoded_audio'
    convert_audio_files(input_folder, output_folder)

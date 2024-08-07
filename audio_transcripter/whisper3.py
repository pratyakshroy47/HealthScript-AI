import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import numpy as np
from pyannote.audio import Pipeline
import torchaudio
import torchaudio.transforms as T
import librosa
from pathlib import Path
import logging
import time
import noisereduce as nr

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

device = "cpu"
torch_dtype = torch.float32

# Set up the diarization pipeline
HF_TOKEN = "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"
diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=HF_TOKEN)
diarization_pipeline.to(torch.device(device))

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    "openai/whisper-large-v3", torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)
processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

def format_timestamp(seconds: float, always_include_hours: bool = False, decimal_marker: str = "."):
    milliseconds = round(seconds * 1000.0)
    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000
    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000
    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000
    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"

def format_as_transcription(raw_segments):
    return "\n".join(
        [
            f"{chunk['speaker']} [{format_timestamp(chunk['timestamp'][0])} -> {format_timestamp(chunk['timestamp'][1])}] {chunk['text']}"
            for chunk in raw_segments
        ]
    )

def align(chunks, segments, group_by_speaker=True):
    new_segments = []
    prev_segment = cur_segment = segments[0]

    for i in range(1, len(segments)):
        cur_segment = segments[i]
        if cur_segment["label"] != prev_segment["label"] and i < len(segments):
            new_segments.append(
                {
                    "segment": {"start": prev_segment["segment"]["start"], "end": cur_segment["segment"]["start"]},
                    "speaker": prev_segment["label"],
                }
            )
            prev_segment = segments[i]

    new_segments.append(
        {
            "segment": {"start": prev_segment["segment"]["start"], "end": cur_segment["segment"]["end"]},
            "speaker": prev_segment["label"],
        }
    )

    end_timestamps = np.array([chunk["timestamp"][-1] for chunk in chunks if chunk["timestamp"][-1] is not None])
    segmented_preds = []
    for segment in new_segments:
        if len(chunks) == 0:
            break
        end_time = segment["segment"]["end"]
        upto_idx = np.argmin(np.abs(end_timestamps - end_time))
        segmented_preds.append(
            {
                "speaker": segment["speaker"],
                "text": "".join([chunk["text"] for chunk in chunks[: upto_idx + 1]]),
                "timestamp": (chunks[0]["timestamp"][0], chunks[upto_idx]["timestamp"][1]),
            }
        )
        chunks = chunks[upto_idx + 1 :]
        end_timestamps = end_timestamps[upto_idx + 1 :]

    return format_as_transcription(segmented_preds)

def remove_noise(audio, sample_rate):
    audio_np = audio.squeeze().numpy()
    reduced_noise = nr.reduce_noise(y=audio_np, sr=sample_rate)
    return torch.tensor(reduced_noise).unsqueeze(0)

def remove_silence(audio, sample_rate):
    vad = T.Vad(sample_rate=sample_rate)
    return vad(audio)

def transcribe(audio_path, task="transcribe", group_by_speaker=True):
    try:
        audio_og, sr = torchaudio.load(audio_path)

        # Noise removal
        audio_denoised = remove_noise(audio_og, sr)

        # Silence removal
        audio_processed = remove_silence(audio_denoised, sr)

        # Resample audio to 16 kHz
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        waveform_resampled = resampler(torch.mean(audio_processed, dim=0)).reshape(1, -1)

        # Get transcription from Whisper
        transcription = pipe(waveform_resampled[-1].numpy(), generate_kwargs={"language": "english", "task": task})

        # Perform diarization
        diarization = diarization_pipeline({"waveform": waveform_resampled, "sample_rate": 16000})
        segments = diarization.to_lab()
        all_segs = []

        for seg in segments.split("\n"):
            if seg.strip():
                start, end, track = seg.split(" ")
                all_segs.append({"segment": {"start": float(start), "end": float(end)}, "label": track})

        return align(transcription["chunks"], all_segs, group_by_speaker)
    except Exception as e:
        print(f'Exception occurred while processing {audio_path}: {e}')
        return None

def run_pipeline_from_local(input_folder, output_folder):
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    for audio_file in Path(input_folder).rglob('*'):
        if audio_file.suffix.lower() in ['.wav', '.mp3', '.webm']:
            print(f'Processing: {audio_file}')
            try:
                output = transcribe(str(audio_file), task="transcribe")
                if output is not None:  # Only write to file if output is not None
                    output_file = Path(output_folder) / (audio_file.stem + '.txt')
                    with open(output_file, 'w') as f:
                        f.write(output)
                    print(f'Transcription saved to: {output_file}')
                else:
                    print(f'No transcription for {audio_file}')
            except Exception as e:
                print(f'Exception occurred: {e}')

# Example usage
input_folder = '/home/ext_pratyakshroy47_gmail_com/medical-fhir/medical_pipeline/audio_transcripter/transcoded_audio'
output_folder = '/home/ext_pratyakshroy47_gmail_com/medical-fhir/medical_pipeline/audio_transcripter/transcript_files'
run_pipeline_from_local(input_folder, output_folder)

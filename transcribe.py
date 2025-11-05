import os
import whisper
import torch
import torchaudio
from pathlib import Path
import numpy as np
from typing import List, Dict, Tuple
from scipy.io import wavfile
import wave
import contextlib
from pydub import AudioSegment
from pydub.silence import split_on_silence


def load_whisper_model(model_name: str = "medium"):
    """Load Whisper model"""
    print(f"Loading Whisper {model_name} model...")
    return whisper.load_model(model_name)


def convert_to_wav(audio_path: str) -> str:
    """Convert audio file to WAV format if needed"""
    if audio_path.lower().endswith('.wav'):
        return audio_path

    print("Converting audio to WAV format...")
    sound = AudioSegment.from_file(audio_path)
    wav_path = os.path.splitext(audio_path)[0] + '.wav'
    sound.export(wav_path, format="wav")
    return wav_path


def get_audio_duration(audio_path: str) -> float:
    """Get duration of audio file in seconds"""
    with contextlib.closing(wave.open(audio_path, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        return frames / float(rate)


def split_audio_on_silence(audio_path: str, min_silence_len: int = 1000, silence_thresh: int = -40) -> List[dict]:
    """Split audio into segments based on silence"""
    print("Segmenting audio based on silence...")
    audio = AudioSegment.from_wav(audio_path)

    # Split on silence
    segments = split_on_silence(
        audio,
        min_silence_len=min_silence_len,  # Minimum silence length in ms
        silence_thresh=silence_thresh,  # Silence threshold in dB
        keep_silence=500  # Keep some silence at the beginning/end of each segment
    )

    # Create time segments
    time_segments = []
    start_time = 0

    for i, segment in enumerate(segments):
        duration = len(segment) / 1000  # Convert to seconds
        if duration < 0.5:  # Skip very short segments
            start_time += duration
            continue

        time_segments.append({
            'start': start_time,
            'end': start_time + duration,
            'segment': segment
        })
        start_time += duration

    return time_segments


def transcribe_with_speakers(audio_path: str, model_name: str = "large"):
    """
    Transcribe an audio file with basic speaker diarization.

    Args:
        audio_path (str): Path to the audio file
        model_name (str): Whisper model to use (tiny, base, small, medium, large)
    """
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found at {audio_path}")
        return

    # Convert to WAV if needed
    wav_path = convert_to_wav(audio_path)

    # Load Whisper model
    model = load_whisper_model(model_name)

    # Split audio into segments
    segments = split_audio_on_silence(wav_path)

    print(f"Found {len(segments)} segments for processing...")

    # Process each segment
    transcriptions = []
    for i, segment in enumerate(segments):
        print(f"Processing segment {i+1}/{len(segments)}...")

        # Export segment to temporary WAV file
        temp_wav = f"temp_segment_{i}.wav"
        segment['segment'].export(temp_wav, format="wav")

        # Transcribe the segment
        result = model.transcribe(temp_wav, language="en")
        text = result["text"].strip()

        if text:
            transcriptions.append({
                'start': segment['start'],
                'end': segment['end'],
                'speaker': "A" if i % 2 == 0 else "B",  # Alternate speakers for now
                'text': text
            })

        # Clean up
        if os.path.exists(temp_wav):
            os.remove(temp_wav)

    # Clean up temporary WAV if we created one
    if wav_path != audio_path and os.path.exists(wav_path):
        os.remove(wav_path)

    # Generate formatted output
    output = []
    current_speaker = None

    for segment in transcriptions:
        if segment['speaker'] != current_speaker:
            output.append(f"\n[Speaker {segment['speaker']}]: {segment['text']}")
            current_speaker = segment['speaker']
        else:
            output.append(segment['text'])

    # Join all text
    full_transcript = ' '.join(output).strip()

    # Create output filename
    output_path = Path(audio_path).with_suffix('.txt')

    # Save transcription
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(full_transcript)

    print(f"\nTranscription complete! Saved to {output_path}")
    print("\nTranscription with Speaker Identification:")
    print("-" * 70)
    print(full_transcript)
    print("-" * 70)

    return full_transcript


def main():
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Look for audio files in the script directory
    audio_extensions = ['.mp3', '.wav', '.m4a', '.ogg', '.flac']
    audio_files = [
        f for f in os.listdir(script_dir)
        if os.path.splitext(f)[1].lower() in audio_extensions
    ]

    if not audio_files:
        print("No audio files found in the script directory.")
        print(f"Supported formats: {', '.join(audio_extensions)}")
        return

    # Use the first audio file found
    audio_file = audio_files[0]
    audio_path = os.path.join(script_dir, audio_file)
    print(f"Found audio file: {audio_file}")

    # Install pydub if not already installed
    try:
        from pydub import AudioSegment
    except ImportError:
        print("Installing pydub for audio processing...")
        os.system("pip install pydub")
        from pydub import AudioSegment

    # Run transcription with speaker diarization
    transcribe_with_speakers(audio_path, model_name="large")


if __name__ == "__main__":
    main()
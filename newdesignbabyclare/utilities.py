from base64 import b64encode
from io import BytesIO
from pathlib import Path
from typing import List, Dict
from typing import Any, Iterator
from urllib.request import urlretrieve
from pydub import AudioSegment
import csv

def download_file(url: str) -> Path:
    example_dirpath = Path(__file__).parent
    data_dirpath = example_dirpath / "data"
    data_dirpath.mkdir(exist_ok=True)
    filepath = data_dirpath / Path(url).name

    urlretrieve(url, filepath)
    return filepath

def generate_audio_stream(filepath: Path, chunk_size: int = 2000) -> Iterator[AudioSegment]:
    segment = AudioSegment.from_file(filepath)
    chunk_count = 0
    while True:
        start_time = chunk_count * chunk_size
        end_time = start_time + chunk_size
        if start_time > len(segment):
            return
        yield segment[start_time:end_time]
        chunk_count += 1

def encode_audio(segment: AudioSegment) -> str:
    bytes_io = BytesIO()
    segment.export(bytes_io, format="wav")
    return b64encode(bytes_io.read()).decode('utf-8')

def print_emotions(emotions: List[Dict[str, Any]]) -> None:
    emotion_map = {e["name"]: e["score"] for e in emotions}
    for emotion in ["Joy", "Sadness", "Anger"]:
        print(f"- {emotion}: {emotion_map[emotion]:.4f}")

def write_csv(filename, data):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Type", "Emotion", "Score"])
        writer.writerows(data)

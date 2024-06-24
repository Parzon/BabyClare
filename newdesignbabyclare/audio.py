import sounddevice as sd
import wavio as wv
import numpy as np
import traceback
from datetime import datetime
from scipy.spatial.distance import cosine
from hume import HumeStreamClient
from hume.models.config import BurstConfig, ProsodyConfig
from utilities import encode_audio, generate_audio_stream, print_emotions
import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Retrieve Hume API key from environment variables
HUME_API_KEY = os.getenv("HUME_API_KEY")

# Load pretrained speaker recognition model
class SpeakerNet(torch.nn.Module):
    def __init__(self):
        super(SpeakerNet, self).__init__()
        self.melspec = MelSpectrogram()
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 512)  # Embedding size

    def forward(self, x):
        x = self.melspec(x)
        x = x.unsqueeze(1)
        x = self.model(x)
        return x

model = SpeakerNet()
model.load_state_dict(torch.load('path_to_pretrained_model.pth'))
model.eval()

def listen_to_user():
    duration = 7  # seconds
    freq = 44100  # sampling frequency
    silence_threshold = 3  # seconds of silence to consider as end of speech

    def save_audio(filename, data, fs):
        wv.write(filename, data, fs, sampwidth=2)

    print("Listening...")
    while True:
        recording = sd.rec(int(duration * freq), samplerate=freq, channels=2)
        sd.wait()  # Wait until the recording is finished
        save_audio("audio.wav", recording, freq)

        if is_silence(recording, silence_threshold):
            break

def is_silence(audio_data, threshold):
    amplitude = np.abs(audio_data)
    max_amplitude = np.max(amplitude)
    if max_amplitude < 0.01:  # Adjust threshold as needed
        return True
    return False

def transcribe_audio(filepath, client):
    audio_file = open(filepath, "rb")
    transcription = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        response_format="text"
    )
    return transcription['text']

def generate_voice_embeddings(audio):
    audio_tensor = torch.tensor(audio, dtype=torch.float32)
    embedding = model(audio_tensor).detach().numpy()
    return embedding.flatten()

async def analyze_voice_sentiment(filepath, csv_data):
    try:
        hume_client = HumeStreamClient(HUME_API_KEY)
        burst_config = BurstConfig()
        prosody_config = ProsodyConfig()
        async with hume_client.connect([burst_config, prosody_config]) as socket:
            for sample_number, audio_sample in enumerate(generate_audio_stream(filepath)):
                encoded_sample = encode_audio(audio_sample)
                await socket.reset_stream()
                result = await socket.send_bytes(encoded_sample)
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(f"\nStreaming sample {sample_number + 1}")

                print("Speech prosody:")
                if "warning" in result["prosody"]:
                    print(result["prosody"]["warning"])
                else:
                    emotions = result["prosody"]["predictions"][0]["emotions"]
                    for emotion in emotions:
                        csv_data.append([timestamp, "Voice", emotion["name"], emotion["score"]])
                    print_emotions(emotions)

                print("Vocal burst")
                if "warning" in result["burst"]:
                    print(result["burst"]["warning"])
                else:
                    emotions = result["burst"]["predictions"][0]["emotions"]
                    for emotion in emotions:
                        csv_data.append([timestamp, "Voice", emotion["name"], emotion["score"]])
                    print_emotions(emotions)
    except Exception:
        print(traceback.format_exc())

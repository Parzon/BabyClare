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
import os
from env import get_hume_api_key
import timm
import torch.nn.functional as F

# Retrieve Hume API key from environment variables
HUME_API_KEY = get_hume_api_key()

# Load pretrained speaker recognition model using timm
class SpeakerNet(torch.nn.Module):
    def __init__(self):
        super(SpeakerNet, self).__init__()
        self.melspec = MelSpectrogram()
        self.model = timm.create_model('resnet34', pretrained=True)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 512)  # Embedding size

    def forward(self, x):
        x = self.melspec(x)
        if x.dim() == 3:  # Ensure the tensor has the right number of dimensions
            x = x.unsqueeze(1)
        x = x.repeat(1, 3, 1, 1)  # (batch_size, 3, n_mels, time_steps) for 3 input channels
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        x = self.model(x)
        return x

model = SpeakerNet()
model.eval()  # Set the model to evaluation mode

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
    return max_amplitude < 0.01  # Adjust threshold as needed

def transcribe_audio(filepath, client):
    with open(filepath, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text"
        )
    return transcription['text']

def generate_voice_embeddings(audio):
    audio_tensor = torch.tensor(audio, dtype=torch.float32)
    audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(1)  # Add batch and channel dimensions

    mel_spec = MelSpectrogram()(audio_tensor)
    if mel_spec.dim() == 3:  # Ensure mel_spec has the shape (batch_size, 1, n_mels, time_steps)
        mel_spec = mel_spec.unsqueeze(1)
    mel_spec = mel_spec.repeat(1, 3, 1, 1)  # Repeat channels
    mel_spec = F.interpolate(mel_spec, size=(224, 224), mode='bilinear', align_corners=False)

    embedding = model(mel_spec).detach().numpy()
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

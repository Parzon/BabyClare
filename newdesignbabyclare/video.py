import cv2 
import traceback
import numpy as np
from scipy.spatial.distance import cosine
from datetime import datetime
from hume import HumeStreamClient
from hume.models.config import FaceConfig
from utilities import print_emotions, encode_image
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from keras.preprocessing import image
from database_operations import find_closest_face, add_user, find_closest_voice
import asyncio
import os
from dotenv import load_dotenv
from audio import generate_voice_embeddings
from env import get_hume_api_key

# Load environment variables from .env file
load_dotenv()

# Retrieve Hume API key from environment variables
HUME_API_KEY = get_hume_api_key()

async def stream_video():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video capture device.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from video capture device.")
            break
        yield frame

    cap.release()

async def analyze_face_sentiment(stream, csv_data):
    try:
        hume_client = HumeStreamClient(HUME_API_KEY)
        config = FaceConfig(identify_faces=True)
        async with hume_client.connect([config]) as socket:
            async for frame in stream:
                encoded_frame = encode_image(frame)
                result = await socket.send_bytes(encoded_frame)
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                emotions = result["face"]["predictions"][0]["emotions"]
                for emotion in emotions:
                    csv_data.append([timestamp, "Face", emotion["name"], emotion["score"]])
                print_emotions(emotions)
    except Exception:
        print(traceback.format_exc())

model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

def generate_face_embeddings(face):
    face = image.img_to_array(face)
    face = np.expand_dims(face, axis=0)
    face = preprocess_input(face)
    embedding = model.predict(face)
    return embedding.flatten()

def detect_faces(frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

def handle_new_face(face_embedding, db):
    closest_face = find_closest_face(face_embedding)
    
    if closest_face:
        name, user_id = closest_face
        print(f"Recognized user {name} with ID {user_id}")
        return name, user_id
    else:
        # First-time user
        name = input("I don't recognize you. What's your name? ")
        user_id = add_user(name, face_embedding, np.zeros(128))  # Placeholder for voice embedding
        print(f"Stored new user {name} with ID {user_id}")
        return name, user_id

def handle_new_voice(voice_embedding, db, user_id):
    closest_voice = find_closest_voice(voice_embedding)
    
    if closest_voice:
        name, closest_user_id = closest_voice
        if closest_user_id == user_id:
            print("Voice matches the detected face.")
        else:
            print("Voice does not match the detected face. You sound different today.")
        return name, closest_user_id
    else:
        # New voice data
        add_user(name, np.zeros(128), voice_embedding)  # Placeholder for face embedding
        print("Stored new voice data.")
        return name, user_id

if __name__ == "__main__":
    csv_data = []

    stream = stream_video()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(analyze_face_sentiment(stream, csv_data))

    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    faces = detect_faces(frame)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face_embedding = generate_face_embeddings(face)
        name, user_id = handle_new_face(face_embedding)

        # Capture voice input and generate embeddings
        listen_to_user()
        audio_file = "audio.wav"
        voice_embedding = generate_voice_embeddings(audio_file)
        handle_new_voice(voice_embedding, user_id)

    cap.release()

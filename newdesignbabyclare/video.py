import cv2 
import traceback
import numpy as np
from scipy.spatial import distance
from datetime import datetime
from hume import HumeStreamClient
from hume.models.config import FaceConfig
from utilities import print_emotions, encode_image
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from keras.preprocessing import image
from scipy.spatial.distance import cosine

# Initialize Hume API key
HUME_API_KEY = "YOUR_HUME_API_KEY"

# Initialize VGGFace model
model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

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

def generate_face_embeddings(face):
    img = image.img_to_array(face)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    embedding = model.predict(img)
    return embedding.flatten()

def detect_faces(frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

def identify_faces(face_embedding, db, threshold=0.5):
    for user in db:
        stored_embedding = user['face_embedding']
        if 1 - cosine(face_embedding, stored_embedding) > threshold:  # Cosine similarity
            return user['name'], user['id']
    return None, None

def store_new_face_data(name, face_embedding, db):
    user_id = len(db) + 1
    db.append({'id': user_id, 'name': name, 'face_embedding': face_embedding})
    return user_id

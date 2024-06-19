# video.py
import cv2  # For capturing video
import traceback
import numpy as np
import dlib
from scipy.spatial import distance
from datetime import datetime
from hume import HumeStreamClient
from hume.models.config import FaceConfig
from utilities import print_emotions
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from keras.preprocessing import image
import numpy as np

HUME_API_KEY = "YOUR_HUME_API_KEY"

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

def encode_image(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    return b64encode(buffer).decode('utf-8')

async def analyze_face_sentiment(stream, csv_data, client):
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

#pip install keras-vggface keras-applications

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

def generate_face_embeddings(face):
    # Placeholder for actual embedding generation using a pre-trained model like FaceNet or dlib
    # For example, using dlib:
    # face_recognition_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
    # shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    # shape = shape_predictor(face, dlib.rectangle(0, 0, face.shape[1], face.shape[0]))
    # embedding = face_recognition_model.compute_face_descriptor(face, shape)
    embedding = np.random.rand(128)  # Placeholder for actual embedding
    return embedding


def identify_faces(face_embedding, db):
    for user in db:
        stored_embedding = user['face_embedding']
        if distance.euclidean(face_embedding, stored_embedding) < 0.6:  # Placeholder threshold
            return user['name'], user['id']
    return None, None

def store_new_face_data(name, face_embedding, db):
    user_id = len(db) + 1
    db.append({'id': user_id, 'name': name, 'face_embedding': face_embedding})
    return user_id

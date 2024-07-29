# background_recognition.py

import cv2
from pathlib import Path
from openai import OpenAI
import base64
import logging

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("application.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def capture_background_image():
    """
    Capture an image from the webcam to analyze the user's background.
    """
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if ret:
        file_path = Path(__file__).parent / "background.jpg"
        cv2.imwrite(str(file_path), frame)
        logger.info(f"Background image saved to {file_path}")
        return file_path
    else:
        logger.error("Failed to capture background image.")
        return None

def encode_image(image_path):
    """
    Encode the image in base64 format for OpenAI API consumption.
    """
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    logger.info(f"Image encoded successfully.")
    return encoded_string

def analyze_background(image_path, client):
    """
    Use OpenAI to analyze the background and generate a description.
    """
    encoded_image = encode_image(image_path)
    prompt = (
        "Describe the scene and background in the following image. "
        "Provide details about the setting, objects, and overall atmosphere. "
        "Respond in a way that would be suitable for generating a conversational response."
    )
    response = client.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an intelligent assistant capable of analyzing images."},
            {"role": "user", "content": f"Image: {encoded_image}\n\n{prompt}"}
        ]
    )
    logger.info("Background analysis completed.")
    return response['choices'][0]['message']['content']

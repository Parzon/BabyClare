import logging
from openai import OpenAI
import time
import asyncio
from utilities import write_csv
from audio import listen_to_user, analyze_voice_sentiment, transcribe_audio, generate_voice_embeddings
from video import capture_video_frames, analyze_face_sentiment, detect_faces, generate_face_embeddings, handle_new_face, handle_new_voice
from sentiment_analysis import wants_response, generate_response, generate_speech_response
from database_operations import setup_database, add_interaction, retrieve_interactions
from background_recognition import capture_background_image, analyze_background

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key="YOUR_OPENAI_API_KEY")

async def process_audio_and_video(csv_data):
    listen_to_user()

    text_result = transcribe_audio("audio.wav", client)
    logger.info(f"Transcribed text: {text_result}")

    time.sleep(10)

    voice_task = asyncio.create_task(analyze_voice_sentiment("audio.wav", csv_data))
    video_stream = capture_video_frames()
    face_task = asyncio.create_task(analyze_face_sentiment(video_stream, csv_data))

    await asyncio.gather(voice_task, face_task)

    return text_result

def analyze_background_and_generate_response(text_result, csv_data, user_id):
    background_image_path = capture_background_image()
    background_info = analyze_background(background_image_path, client)
    logger.info(f"Background info: {background_info}")

    interactions = retrieve_interactions(user_id)

    if wants_response(text_result, client):
        response_text = generate_response(text_result, "Sentiment analysis data", client)
        response_text += f"\n\nAlso, I noticed that you're in a {background_info}."
        logger.info(f"Generated response: {response_text}")
        speech_file = generate_speech_response(response_text, client)
        logger.info(f"Speech response saved to {speech_file}")

        add_interaction(user_id, text_result, response_text, "Voice Sentiment Data", "Face Sentiment Data")
    else:
        logger.info("User did not ask for a response. Continuing to listen...")
        listen_to_user()

    write_csv("sentiment_analysis.csv", csv_data)
    logger.info("Data written to CSV")

async def main():
    csv_data = []

    logger.info("Starting application")

    setup_database()

    # Analyze video stream for face sentiment and recognition
    stream = capture_video_frames()
    loop = asyncio.get_event_loop()
    face_task = asyncio.create_task(analyze_face_sentiment(stream, csv_data))

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

        text_result = transcribe_audio(audio_file, client)
        logger.info(f"Transcribed text: {text_result}")

        await asyncio.gather(face_task)

        if wants_response(text_result, client):
            response_text = generate_response(text_result, "Sentiment analysis data", client)
            logger.info(f"Generated response: {response_text}")
            speech_file = generate_speech_response(response_text, client)
            logger.info(f"Speech response saved to {speech_file}")
        else:
            logger.info("User did not ask for a response. Continuing to listen...")
            listen_to_user()

        analyze_background_and_generate_response(text_result, csv_data, user_id)

    write_csv("sentiment_analysis.csv", csv_data)
    logger.info("Data written to CSV")

if __name__ == "__main__":
    asyncio.run(main())

import logging
from openai import OpenAI
import time
import asyncio
from utilities import write_csv
from audio import capture_audio, analyze_voice_sentiment, save_audio, transcribe_audio
from video import capture_video_frames, detect_faces, analyze_face_sentiment
from sentiment_analysis import wants_response, generate_response, generate_speech_response

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = OpenAI(api_key="YOUR_OPENAI_API_KEY")

if __name__ == "__main__":
    csv_data = []

    logger.info("Starting application")

    listen_to_user()

    text_result = transcribe_audio("audio.wav", client)
    logger.info(f"Transcribed text: {text_result}")

    time.sleep(10)

    voice_task = asyncio.create_task(analyze_voice_sentiment("audio.wav", csv_data))
    video_stream = capture_video_frames()
    face_task = asyncio.create_task(analyze_face_sentiment(video_stream, csv_data))

    asyncio.run(asyncio.gather(voice_task, face_task))

    if wants_response(text_result, client):
        response_text = generate_response(text_result, "Sentiment analysis data", client)
        logger.info(f"Generated response: {response_text}")
        speech_file = generate_speech_response(response_text, client)
        logger.info(f"Speech response saved to {speech_file}")
    else:
        logger.info("User did not ask for a response. Continuing to listen...")
        listen_to_user()

    write_csv("sentiment_analysis.csv", csv_data)
    logger.info("Data written to CSV")

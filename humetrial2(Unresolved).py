import asyncio
import cv2
import numpy as np
import sounddevice as sd
from hume import HumeStreamClient
from hume.models.config import LanguageConfig, FaceConfig

def setup_audio():
    stream = sd.InputStream(samplerate=44100, channels=1, dtype='int16')
    stream.start()
    return stream

def setup_video():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Could not open video source")
    return cap

async def collect_data(audio_stream, video_capture):
    voice_client = HumeStreamClient(api_key='')
    face_client = HumeStreamClient(api_key='')
    voice_config = LanguageConfig()
    face_config = FaceConfig(identify_faces=True)

    try:
        async with voice_client.connect([voice_config]) as voice_socket, \
                  face_client.connect([face_config]) as face_socket:
            while True:
                ret, frame = video_capture.read()
                if not ret:
                    print("Failed to fetch video frame")
                    continue

                _, encoded_image = cv2.imencode('.jpg', frame)
                video_data = encoded_image.tobytes()

                audio_data = audio_stream.read(1024)[0]
                audio_bytes = np.frombuffer(audio_data, dtype=np.int16).tobytes()

                try:
                    await voice_socket.send_bytes(audio_bytes)
                    await face_socket.send_bytes(video_data)
                    print("Data sent successfully.")
                except Exception as e:
                    print("Error sending data:", e)

                await asyncio.sleep(0.1)
    except Exception as e:
        print("Unexpected error occurred:", e)

def close_streams(stream, video):
    video.release()
    stream.stop()

def main():
    audio_stream = setup_audio()
    video_capture = setup_video()
    try:
        asyncio.run(collect_data(audio_stream, video_capture))
    except Exception as e:
        print("Error during operation:", e)
    finally:
        close_streams(audio_stream, video_capture)

if __name__ == "__main__":
    main()

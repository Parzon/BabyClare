import asyncio
import cv2
import numpy as np
import pandas as pd
import base64
from datetime import datetime
from hume import HumeStreamClient
from hume.models.config import ProsodyConfig, FaceConfig
import sounddevice as sd
import queue

# Replace with your Hume API key
api_key = "YOUR_API_KEY"

# Queue for audio processing
audio_queue = queue.Queue()

# Callback for audio processing, logs any errors and enqueues audio data with timestamp
def audio_callback(indata, frames, time, status):
    if status:
        print(f"Audio status: {status}")
    audio_queue.put((datetime.now(), indata.copy()))

# Async function to process audio, logs processing steps and errors
async def process_audio(client, audio_config, runtime):
    start_time = datetime.now()
    audio_results = []
    while (datetime.now() - start_time).total_seconds() < runtime:
        if not audio_queue.empty():
            timestamp, data = audio_queue.get()
            print(f"Processing audio data at {timestamp}")
            encoded_audio = base64.b64encode(data).decode('utf-8')
            try:
                async with client.connect([audio_config]) as socket:
                    response = await socket.send_bytes(base64.b64decode(encoded_audio))
                    print("Audio analysis result:", response)
                    audio_results.append({"timestamp": timestamp, "data": response})
            except Exception as e:
                print(f"Error processing audio data: {e}")
        await asyncio.sleep(0.1)
    return pd.DataFrame(audio_results)

# Async function to capture and process video frames, logs each step
async def process_video(client, video_config, runtime):
    start_time = datetime.now()
    video_capture = cv2.VideoCapture(0)
    video_results = []

    while (datetime.now() - start_time).total_seconds() < runtime:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to capture video frame.")
            break
        _, buffer = cv2.imencode('.jpg', frame)
        encoded_video = base64.b64encode(buffer)
        timestamp = datetime.now()
        try:
            async with client.connect([video_config]) as socket:
                response = await socket.send_bytes(encoded_video)
                print("Video frame analysis result:", response)
                video_results.append({"timestamp": timestamp, "data": response})
        except Exception as e:
            print(f"Error processing video data: {e}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    return pd.DataFrame(video_results)

# Main async function to run both audio and video processing concurrently
async def main():
    client = HumeStreamClient(api_key)
    audio_config = ProsodyConfig()
    video_config = FaceConfig(identify_faces=True)
    runtime = 60  # Runtime in seconds

    audio_task = asyncio.create_task(process_audio(client, audio_config, runtime))
    video_task = asyncio.create_task(process_video(client, video_config, runtime))

    audio_results, video_results = await asyncio.gather(audio_task, video_task)

    audio_results.to_csv('audio_results.csv', index=False)
    video_results.to_csv('video_results.csv', index=False)

    print("Audio Results Dataframe:", audio_results)
    print("Video Results Dataframe:", video_results)

if __name__ == '__main__':
    asyncio.run(main())

import asyncio
import cv2
import os
import base64
from hume import HumeStreamClient
from hume.models.config import FaceConfig

api_key = 'sQp9AtmP52EQ5kD1AG9aQxYfZgrkPvwrOKGZZAxaZAqbynvv'  # Replace with your Hume AI API Key
frames_directory = './captured_frames'  # Path to the directory containing frames

# Function to encode image to base64 and convert it to bytes
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read())

# Async function to send a frame to Hume API and get the result
async def analyze_frame(frame_path, client):
    encoded_image = encode_image_to_base64(frame_path)
    config = FaceConfig(identify_faces=True)  # Customize this as per your Hume AI setup
    async with client.connect([config]) as socket:
        response = await socket.send_bytes(encoded_image)  # Now sending bytes directly
        return response

# Main function to process and send all frames
async def process_frames():
    client = HumeStreamClient(api_key)
    results = []

    for frame_filename in sorted(os.listdir(frames_directory)):
        frame_path = os.path.join(frames_directory, frame_filename)
        if frame_filename.endswith(".jpg"):  # Ensure to process only JPG images
            print(f"Analyzing {frame_path}...")
            result = await analyze_frame(frame_path, client)
            results.append((frame_path, result))
            print(f"Result for {frame_path}: {result}")

    return results

# Run the async main function
if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    final_results = loop.run_until_complete(process_frames())
    print("Analysis Complete. Results:")
    for frame_result in final_results:
        print(frame_result)

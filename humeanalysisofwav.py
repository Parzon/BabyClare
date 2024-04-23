import asyncio
from hume import HumeStreamClient
from hume.models.config import ProsodyConfig

 # Your API key here
api_key = ''
# Filepath to your audio file
audio_file_path = "output.wav" 

# Main function to send audio data
async def main():
    client = HumeStreamClient(api_key)
    config = ProsodyConfig()
    
    async with client.connect([config]) as socket:
        # The send_file method is used directly with the file path
        result = await socket.send_file(audio_file_path)
        print(result)

# Run the main function
if __name__ == '__main__':
    asyncio.run(main())

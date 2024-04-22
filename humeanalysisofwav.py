import asyncio
from hume import HumeStreamClient
from hume.models.config import ProsodyConfig

# Your Hume.ai API Key
api_key = 'sQp9AtmP52EQ5kD1AG9aQxYfZgrkPvwrOKGZZAxaZAqbynvv'
# Filepath to your audio file
audio_file_path = "output.wav"  # Ensure you put the correct file path here

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

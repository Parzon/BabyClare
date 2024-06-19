def wants_response(text, client):
    response = client.chat.completions.create(
        model="gpt-4",
        max_tokens=5,
        messages=[
            {"role": "system", "content": "You are a deterministic virtual assistant."},
            {"role": "user", "content": f"Analyze the following text: {text}, determine if the user wants a response or not. Some responses that will trigger a yes sound like 'what do you think', 'what do you feel', 'tell me'. Response: Yes or No"}
        ]
    )
    if response['choices'][0]['message']['content'].strip().lower() == "yes":
        return True
    return False

def generate_response(text, sentiment, client):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a friendly virtual companion."},
            {"role": "user", "content": f"Analyze the following text and sentiment: {text}. Sentiment: {sentiment}"}
        ]
    )
    return response['choices'][0]['message']['content']

def generate_speech_response(text, client):
    speech_file_path = Path(__file__).parent / "response_speech.mp3"
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text
    )
    response.stream_to_file(speech_file_path)
    return speech_file_path

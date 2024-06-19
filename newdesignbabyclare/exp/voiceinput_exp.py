#this is the new deisgn for the baby clare project
#this is the voice input file
#this file is responsible for taking in the voice input from the user
#and converting it to text using the openai whisper api with this synatax
""" 
from openai import OpenAI
client = OpenAI()

audio_file= open("/path/to/file/audio.mp3", "rb")
transcription = client.audio.transcriptions.create(
  model="whisper-1", 
  file=audio_file
  response_format="text"
)
print(transcription.text)
"""
#this project resembles a young child speaking to people 
#it doesnt immediately end the conversation when the user stops speaking
#it waits for a few seconds before ending the conversation
#this is to make the conversation feel more natural
#its for old people so they talk to it slowly or theyre probably hard of hearing 
#so it waits for a few seconds before ending the conversation
#It coverts the voice input to text 
#the voice sentiment needs to be analysed too 
#the voice file is turned into a wav and then sent to the hume.ai api
#that can be done through hume.ai 
#it takes the text, the voice emotion waits a few seconds and returns a response


#below is the high level documentation for the voice input file
"""
Speech to text
Learn how to turn audio into text

Introduction
The Audio API provides two speech to text endpoints, transcriptions and translations, based on our state-of-the-art open source large-v2 Whisper model. They can be used to:

Transcribe audio into whatever language the audio is in.
Translate and transcribe the audio into english.
File uploads are currently limited to 25 MB and the following input file types are supported: mp3, mp4, mpeg, mpga, m4a, wav, and webm.

Quickstart
Transcriptions
The transcriptions API takes as input the audio file you want to transcribe and the desired output file format for the transcription of the audio. We currently support multiple input and output file formats.

Transcribe audio
python

python
from openai import OpenAI
client = OpenAI()

audio_file= open("/path/to/file/audio.mp3", "rb")
transcription = client.audio.transcriptions.create(
  model="whisper-1", 
  file=audio_file
)
print(transcription.text)
By default, the response type will be json with the raw text included.

{
  "text": "Imagine the wildest idea that you've ever had, and you're curious about how it might scale to something that's a 100, a 1,000 times bigger.
....
}
The Audio API also allows you to set additional parameters in a request. For example, if you want to set the response_format as text, your request would look like the following:

Additional options
python

python
from openai import OpenAI
client = OpenAI()

audio_file = open("/path/to/file/speech.mp3", "rb")
transcription = client.audio.transcriptions.create(
  model="whisper-1", 
  file=audio_file, 
  response_format="text"
)
print(transcription.text)
The API Reference includes the full list of available parameters.

Translations
The translations API takes as input the audio file in any of the supported languages and transcribes, if necessary, the audio into English. This differs from our /Transcriptions endpoint since the output is not in the original input language and is instead translated to English text.

Translate audio
python

python
from openai import OpenAI
client = OpenAI()

audio_file= open("/path/to/file/german.mp3", "rb")
translation = client.audio.translations.create(
  model="whisper-1", 
  file=audio_file
)
print(translation.text)
In this case, the inputted audio was german and the outputted text looks like:

Hello, my name is Wolfgang and I come from Germany. Where are you heading today?
We only support translation into English at this time.

Supported languages
We currently support the following languages through both the transcriptions and translations endpoint:

Afrikaans, Arabic, Armenian, Azerbaijani, Belarusian, Bosnian, Bulgarian, Catalan, Chinese, Croatian, Czech, Danish, Dutch, English, Estonian, Finnish, French, Galician, German, Greek, Hebrew, Hindi, Hungarian, Icelandic, Indonesian, Italian, Japanese, Kannada, Kazakh, Korean, Latvian, Lithuanian, Macedonian, Malay, Marathi, Maori, Nepali, Norwegian, Persian, Polish, Portuguese, Romanian, Russian, Serbian, Slovak, Slovenian, Spanish, Swahili, Swedish, Tagalog, Tamil, Thai, Turkish, Ukrainian, Urdu, Vietnamese, and Welsh.

While the underlying model was trained on 98 languages, we only list the languages that exceeded <50% word error rate (WER) which is an industry standard benchmark for speech to text model accuracy. The model will return results for languages not listed above but the quality will be low.

Timestamps
By default, the Whisper API will output a transcript of the provided audio in text. The timestamp_granularities[] parameter enables a more structured and timestamped json output format, with timestamps at the segment, word level, or both. This enables word-level precision for transcripts and video edits, which allows for the removal of specific frames tied to individual words.

Timestamp options
python

python
from openai import OpenAI
client = OpenAI()

audio_file = open("speech.mp3", "rb")
transcript = client.audio.transcriptions.create(
  file=audio_file,
  model="whisper-1",
  response_format="verbose_json",
  timestamp_granularities=["word"]
)

print(transcript.words)
Longer inputs
By default, the Whisper API only supports files that are less than 25 MB. If you have an audio file that is longer than that, you will need to break it up into chunks of 25 MB's or less or used a compressed audio format. To get the best performance, we suggest that you avoid breaking the audio up mid-sentence as this may cause some context to be lost.

One way to handle this is to use the PyDub open source Python package to split the audio:

from pydub import AudioSegment

song = AudioSegment.from_mp3("good_morning.mp3")

# PyDub handles time in milliseconds
ten_minutes = 10 * 60 * 1000

first_10_minutes = song[:ten_minutes]

first_10_minutes.export("good_morning_10.mp3", format="mp3")
OpenAI makes no guarantees about the usability or security of 3rd party software like PyDub.

Prompting
You can use a prompt to improve the quality of the transcripts generated by the Whisper API. The model will try to match the style of the prompt, so it will be more likely to use capitalization and punctuation if the prompt does too. However, the current prompting system is much more limited than our other language models and only provides limited control over the generated audio. Here are some examples of how prompting can help in different scenarios:

Prompts can be very helpful for correcting specific words or acronyms that the model may misrecognize in the audio. For example, the following prompt improves the transcription of the words DALL·E and GPT-3, which were previously written as "GDP 3" and "DALI": "The transcript is about OpenAI which makes technology like DALL·E, GPT-3, and ChatGPT with the hope of one day building an AGI system that benefits all of humanity"
To preserve the context of a file that was split into segments, you can prompt the model with the transcript of the preceding segment. This will make the transcript more accurate, as the model will use the relevant information from the previous audio. The model will only consider the final 224 tokens of the prompt and ignore anything earlier. For multilingual inputs, Whisper uses a custom tokenizer. For English only inputs, it uses the standard GPT-2 tokenizer which are both accessible through the open source Whisper Python package.
Sometimes the model might skip punctuation in the transcript. You can avoid this by using a simple prompt that includes punctuation: "Hello, welcome to my lecture."
The model may also leave out common filler words in the audio. If you want to keep the filler words in your transcript, you can use a prompt that contains them: "Umm, let me think like, hmm... Okay, here's what I'm, like, thinking."
Some languages can be written in different ways, such as simplified or traditional Chinese. The model might not always use the writing style that you want for your transcript by default. You can improve this by using a prompt in your preferred writing style.
Improving reliability
As we explored in the prompting section, one of the most common challenges faced when using Whisper is the model often does not recognize uncommon words or acronyms. To address this, we have highlighted different techniques which improve the reliability of Whisper in these cases:

Using the prompt parameter
The first method involves using the optional prompt parameter to pass a dictionary of the correct spellings.

Since it wasn't trained using instruction-following techniques, Whisper operates more like a base GPT model. It's important to keep in mind that Whisper only considers the first 244 tokens of the prompt.

Prompt parameter
python

python
from openai import OpenAI
client = OpenAI()

audio_file = open("/path/to/file/speech.mp3", "rb")
transcription = client.audio.transcriptions.create(
  model="whisper-1", 
  file=audio_file, 
  response_format="text",
  prompt="ZyntriQix, Digique Plus, CynapseFive, VortiQore V8, EchoNix Array, OrbitalLink Seven, DigiFractal Matrix, PULSE, RAPT, B.R.I.C.K., Q.U.A.R.T.Z., F.L.I.N.T."
)
print(transcription.text)
While it will increase reliability, this technique is limited to only 244 characters so your list of SKUs would need to be relatively small in order for this to be a scalable solution.

Post-processing with GPT-4
The second method involves a post-processing step using GPT-4 or GPT-3.5-Turbo.

We start by providing instructions for GPT-4 through the system_prompt variable. Similar to what we did with the prompt parameter earlier, we can define our company and product names.

Post-processing
python

python
system_prompt = "You are a helpful assistant for the company ZyntriQix. Your task is to correct any spelling discrepancies in the transcribed text. Make sure that the names of the following products are spelled correctly: ZyntriQix, Digique Plus, CynapseFive, VortiQore V8, EchoNix Array, OrbitalLink Seven, DigiFractal Matrix, PULSE, RAPT, B.R.I.C.K., Q.U.A.R.T.Z., F.L.I.N.T. Only add necessary punctuation such as periods, commas, and capitalization, and use only the context provided."

def generate_corrected_transcript(temperature, system_prompt, audio_file):
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=temperature,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": transcribe(audio_file, "")
            }
        ]
    )
    return completion.choices[0].message.content

corrected_text = generate_corrected_transcript(0, system_prompt, fake_company_filepath)
If you try this on your own audio file, you can see that GPT-4 manages to correct many misspellings in the transcript. Due to its larger context window, this method might be more scalable than using Whisper's prompt parameter and is more reliable since GPT-4 can be instructed and guided in ways that aren't possible with Whisper given the lack of instruction following.
"""



#Trial 1 
# this is the new design for the Baby Clare project
# this is the voice input file
# this file is responsible for taking in the voice input from the user
# and converting it to text using the OpenAI Whisper API

# Step 1: Import necessary libraries
from openai import OpenAI
import time

# Step 2: Initialize the OpenAI client
client = OpenAI()

#step 3 should be to take in the audio file from the user, continuously listening to the user until they stop speaking for 3 seconds. but a response should only be given if they say "what do you think or what do you feel or tell me" use llms to figure out if the person wants to hear from the robo or not


# Step 3: Save and Open the audio file at the end of every 3 second gap, but dont genrate response
# pseudo code: 
# audio_file = open("/path/to/file/audio.mp3", "rb")

# Step 4: Keep Converting voice input to text using Whisper API
# pseudo code:
# transcription = client.audio.transcriptions.create(
#     model="whisper-1", 
#     file=audio_file,
#     response_format="text"
# )
# print(transcription.text)

# Step 5: Wait for a few seconds to ensure conversation has ended
# pseudo code:
# time.sleep(10)

# Step 6: Analyze sentiment of the voice using Hume AI API
# pseudo code for Hume AI API integration, check hume documentation for more details, stream the voice file to the hume api
# sentiment_analysis = hume_client.analyze_emotion(
#     file=audio_file,
#     model="emotion"
# )

# Step 7: Process the text and sentiment analysis results
# pseudo code for processing:
# text_result = transcription.text
# sentiment_result = sentiment_analysis['emotion']

# Step 8: Return a response based on the processed input
# pseudo code:
# response = generate_response(text_result, sentiment_result)
# print(response)

# Note: The function 'generate_response' needs to be defined elsewhere in the project
# This function should take in the transcribed text and sentiment analysis result to generate an appropriate response





#Trial 2 
# this is the new design for the Baby Clare project
# this is the voice input file
# this file is responsible for taking in the voice input from the user
# and converting it to text using the OpenAI Whisper API

# Step 1: Import necessary libraries
from openai import OpenAI
import time
import sounddevice as sd
import wavio as wv

# Step 2: Initialize the OpenAI client
client = OpenAI()

# Function to continuously listen to the user until they stop speaking for 3 seconds
def listen_to_user():
    # Set the duration of the recording
    duration = 7  # seconds
    freq = 44100  # sampling frequency
    silence_threshold = 3  # seconds of silence to consider as end of speech

    # Function to save audio file
    def save_audio(filename, data, fs):
        wv.write(filename, data, fs, sampwidth=2)

    # Listen continuously and save audio files
    print("Listening...")
    while True:
        # Record audio
        recording = sd.rec(int(duration * freq), samplerate=freq, channels=2)
        sd.wait()  # Wait until the recording is finished
        save_audio("audio.wav", recording, freq)

        # Analyze the audio for silence (pseudo code, replace with actual silence detection logic)
        if is_silence(recording, silence_threshold):
            break

# Function to check if the recording is silence
def is_silence(audio_data, threshold):
    # Pseudo code for silence detection
    # silence detected if the wav file is flat for the threshold duration
    # Analyze the audio_data and return True if silence is detected for the threshold duration
    pass

# Step 3: Take in the audio file from the user, continuously listening to the user until they stop speaking for 3 seconds
listen_to_user()

# Step 4: Open the audio file
audio_file = open("audio.wav", "rb")

# Step 5: Convert voice input to text using Whisper API
transcription = client.audio.transcriptions.create(
    model="whisper-1", 
    file=audio_file,
    response_format="text"
)
print(transcription.text)

# Step 6: Wait for a few seconds to ensure conversation has ended
time.sleep(10)

# Step 7: Analyze sentiment of the voice using Hume AI API
# Pseudo code for Hume AI API integration, check Hume documentation for more details, stream the voice file to the Hume API
# sentiment_analysis = hume_client.analyze_emotion(
#     file=audio_file,
#     model="emotion"
# )

# Step 8: Process the text and sentiment analysis results
text_result = transcription.text
# sentiment_result = sentiment_analysis['emotion']

# Step 9: Determine if the user wants a response using an LLM
response_trigger_phrases = ["what do you think", "what do you feel", "tell me"]

def wants_response(text):
    # Pseudo code to determine if the user wants a response
    #openai client 
    #chat.completions.create(systemprompt = "Determine if the user wants a response or not, respond with yes or no "
    #token_limit = 5)
    #if response == "yes":
    #    return True

    if wants_response==True:
    # Pseudo code to generate response based on the processed input
    # response = generate_response(text_result, sentiment_result)
    # print(response)
        pass
    else:
# continue listening to the user

        listen_to_user()





#Trial 3
# this is the new design for the Baby Clare project
# this is the voice input file
# this file is responsible for taking in the voice input from the user
# and converting it to text using the OpenAI Whisper API

# Step 1: Import necessary libraries
from openai import OpenAI
import time
import sounddevice as sd
import wavio as wv
import numpy as np
import requests

# Step 2: Initialize the OpenAI client
client = OpenAI(api_key="YOUR_OPENAI_API_KEY")

# Hume AI API details
HUME_API_URL = "https://api.hume.ai/v1/analyze"
HUME_API_KEY = "YOUR_HUME_API_KEY"

# Function to continuously listen to the user until they stop speaking for 3 seconds
def listen_to_user():
    # Set the duration of the recording
    duration = 7  # seconds
    freq = 44100  # sampling frequency
    silence_threshold = 3  # seconds of silence to consider as end of speech

    # Function to save audio file
    def save_audio(filename, data, fs):
        wv.write(filename, data, fs, sampwidth=2)

    # Listen continuously and save audio files
    print("Listening...")
    while True:
        # Record audio
        recording = sd.rec(int(duration * freq), samplerate=freq, channels=2)
        sd.wait()  # Wait until the recording is finished
        save_audio("audio.wav", recording, freq)

        # Analyze the audio for silence
        if is_silence(recording, silence_threshold):
            break

# Function to check if the recording is silence
def is_silence(audio_data, threshold):
    # Calculate the amplitude of the audio data
    amplitude = np.abs(audio_data)
    max_amplitude = np.max(amplitude)
    if max_amplitude < 0.01:  # Adjust threshold as needed
        return True
    return False

# Function to determine if the user wants a response
def wants_response(text):
    response = client.chat.completions.create(
        model="gpt-4",
        max_tokens=5,
        messages=[
            {"role": "system", "content": "You are a deterministic virtual assistant."},
            {"role": "user", "content": f"Analyze the following text: {text}, determine if the user wants a response or not. Some responses that will trigger a yes sound like ""what do you think"", ""what do you feel"", ""tell me"". Response: Yes or No"}
        ]
    )
    if response['choices'][0]['message']['content'].lower() == "yes":
        return True

# Function to analyze sentiment using Hume AI API
def analyze_sentiment(file_path):
    headers = {
        "Authorization": f"Bearer {HUME_API_KEY}",
        "Content-Type": "audio/wav"
    }
    with open(file_path, "rb") as f:
        response = requests.post(HUME_API_URL, headers=headers, data=f)
    if response.status_code == 200:
        return response.json()
    else:
        response.raise_for_status()

# Function to generate a response using OpenAI GPT
def generate_response(text, sentiment):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a friendly virtual companion."},
            {"role": "user", "content": f"Analyze the following text and sentiment: {text}. Sentiment: {sentiment}"}
        ]
    )
    return response['choices'][0]['message']['content']

#funtion to generate a speech response based on the processed input, change this to fit out requiremnts
from pathlib import Path
from openai import OpenAI
client = OpenAI()

speech_file_path = Path(__file__).parent / "speech.mp3"
response = client.audio.speech.create(
  model="tts-1",
  voice="alloy",
  input="Today is a wonderful day to build something people love!"
)

response.stream_to_file(speech_file_path)

# Main logic
if __name__ == "__main__":
    # Step 3: Take in the audio file from the user, continuously listening to the user until they stop speaking for 3 seconds
    listen_to_user()

    # Step 4: Open the audio file
    audio_file = open("audio.wav", "rb")

    # Step 5: Convert voice input to text using Whisper API
    transcription = client.audio.transcriptions.create(
        model="whisper-1", 
        file=audio_file,
        response_format="text"
    )
    text_result = transcription['text']
    print(text_result)

    # Step 6: Wait for a few seconds to ensure conversation has ended
    time.sleep(10)

    # Step 7: Analyze sentiment of the voice using Hume AI API
    sentiment_result = analyze_sentiment("audio.wav")
    print(sentiment_result)

    # Step 9: Determine if the user wants a response
    if wants_response(text_result):
        # Step 8: Generate and print response based on the processed input
        response = generate_response(text_result, sentiment_result)
        print(response)
    else:
        print("User did not ask for a response. Continuing to listen...")
        listen_to_user()





#Trial 4

# this is the new design for the Baby Clare project
# this is the voice input file
# this file is responsible for taking in the voice input from the user
# and converting it to text using the OpenAI Whisper API

# Step 1: Import necessary libraries
from openai import OpenAI
import time
import sounddevice as sd
import wavio as wv
import numpy as np
import requests
from pathlib import Path

# Step 2: Initialize the OpenAI client
client = OpenAI(api_key="YOUR_OPENAI_API_KEY")

# Hume AI API details
HUME_API_URL = "https://api.hume.ai/v1/analyze"
HUME_API_KEY = "YOUR_HUME_API_KEY"

# Function to continuously listen to the user until they stop speaking for 3 seconds
def listen_to_user():
    duration = 7  # seconds
    freq = 44100  # sampling frequency
    silence_threshold = 3  # seconds of silence to consider as end of speech

    def save_audio(filename, data, fs):
        wv.write(filename, data, fs, sampwidth=2)

    print("Listening...")
    while True:
        recording = sd.rec(int(duration * freq), samplerate=freq, channels=2)
        sd.wait()  # Wait until the recording is finished
        save_audio("audio.wav", recording, freq)

        if is_silence(recording, silence_threshold):
            break

def is_silence(audio_data, threshold):
    amplitude = np.abs(audio_data)
    max_amplitude = np.max(amplitude)
    if max_amplitude < 0.01:  # Adjust threshold as needed
        return True
    return False

def wants_response(text):
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

def analyze_sentiment(file_path):
    headers = {
        "Authorization": f"Bearer {HUME_API_KEY}",
        "Content-Type": "audio/wav"
    }
    with open(file_path, "rb") as f:
        response = requests.post(HUME_API_URL, headers=headers, data=f)
    if response.status_code == 200:
        return response.json()
    else:
        response.raise_for_status()

def generate_response(text, sentiment):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a friendly virtual companion."},
            {"role": "user", "content": f"Analyze the following text and sentiment: {text}. Sentiment: {sentiment}"}
        ]
    )
    return response['choices'][0]['message']['content']

def generate_speech_response(text):
    speech_file_path = Path(__file__).parent / "response_speech.mp3"
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text
    )
    response.stream_to_file(speech_file_path)
    return speech_file_path

if __name__ == "__main__":
    listen_to_user()

    audio_file = open("audio.wav", "rb")

    transcription = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        response_format="text"
    )
    text_result = transcription['text']
    print(text_result)

    time.sleep(10)

    sentiment_result = analyze_sentiment("audio.wav")
    print(sentiment_result)

    if wants_response(text_result):
        response_text = generate_response(text_result, sentiment_result)
        print(response_text)
        speech_file = generate_speech_response(response_text)
        print(f"Speech response saved to {speech_file}")
    else:
        print("User did not ask for a response. Continuing to listen...")
        listen_to_user()


"""
Key Components and Workflow

	1.	Libraries and Initialization
	•	We use various libraries to handle audio recording, processing, and API interactions.
	•	OpenAI’s API client is initialized to interact with Whisper (for speech-to-text) and GPT-4 (for text analysis and response generation).
	•	Hume AI’s API is used for sentiment analysis.
	2.	Listening to User Speech
	•	The system continuously listens to the user until there is a 3-second pause, indicating the end of speech.
	•	Audio is recorded in 7-second intervals to ensure we capture ongoing speech without interruptions.
	3.	Saving Audio
	•	The recorded audio is saved as a WAV file (audio.wav) for further processing.
	4.	Silence Detection
	•	The system analyzes the audio to detect if the user has stopped speaking. This is done by checking the amplitude (volume) of the audio signal. If the amplitude is below a certain threshold, it is considered silence.
	5.	Speech-to-Text Conversion
	•	The saved audio file is sent to OpenAI’s Whisper API, which converts the speech into text. This text represents what the user said.
	6.	Sentiment Analysis
	•	The recorded audio file is sent to Hume AI’s API to analyze the emotional tone of the user’s speech. This helps in understanding the user’s emotional state during the conversation.
	7.	Determining if a Response is Needed
	•	The transcribed text is analyzed to determine if the user is expecting a response. Phrases like “what do you think,” “what do you feel,” or “tell me” indicate that a response is needed.
	•	OpenAI’s GPT-4 is used to analyze the text and decide if the user wants a response.
	8.	Generating a Response
	•	If a response is required, GPT-4 generates a text response based on the user’s input and the sentiment analysis.
	•	The generated text is then converted into speech using OpenAI’s Text-to-Speech (TTS) API, creating an audio file (response_speech.mp3) that the
"""

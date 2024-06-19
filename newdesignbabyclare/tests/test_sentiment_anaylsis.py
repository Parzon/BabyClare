# test_sentiment_analysis.py

import unittest
from pathlib import Path
from sentiment_analysis import wants_response, generate_response, generate_speech_response
from openai import OpenAI

client = OpenAI(api_key="YOUR_OPENAI_API_KEY")

class TestSentimentAnalysisFunctions(unittest.TestCase):

    def test_wants_response(self):
        text = "What do you think about this?"
        result = wants_response(text, client)
        self.assertTrue(result)

    def test_generate_response(self):
        text = "Hello"
        sentiment = "Happy"
        response = generate_response(text, sentiment, client)
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)

    def test_generate_speech_response(self):
        text = "Hello"
        speech_file_path = generate_speech_response(text, client)
        self.assertTrue(speech_file_path.exists())
        self.assertGreater(speech_file_path.stat().st_size, 0)

if __name__ == '__main__':
    unittest.main()

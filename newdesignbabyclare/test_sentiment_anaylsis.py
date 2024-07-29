import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
from sentiment_analysis import wants_response, generate_response, generate_speech_response
from openai import OpenAI

client = OpenAI(api_key="YOUR_OPENAI_API_KEY")

class TestSentimentAnalysisFunctions(unittest.TestCase):

    @patch('openai.ChatCompletion.create')
    def test_wants_response(self, mock_create):
        mock_create.return_value = {'choices': [{'message': {'content': 'yes'}}]}
        text = "What do you think about this?"
        result = wants_response(text, client)
        self.assertTrue(result)

    @patch('openai.ChatCompletion.create')
    def test_generate_response(self, mock_create):
        mock_create.return_value = {'choices': [{'message': {'content': 'Response text'}}]}
        text = "Hello"
        sentiment = "Happy"
        response = generate_response(text, sentiment, client)
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)

    @patch('openai.Audio.create')
    def test_generate_speech_response(self, mock_create):
        mock_response = MagicMock()
        mock_response.stream_to_file.return_value = None
        mock_create.return_value = mock_response

        text = "Hello"
        speech_file_path = generate_speech_response(text, client)
        self.assertTrue(speech_file_path.exists())
        self.assertGreater(speech_file_path.stat().st_size, 0)

if __name__ == '__main__':
    unittest.main()

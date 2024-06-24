# test_background_recognition.py

import unittest
from background_recognition import capture_background_image, analyze_background
from openai import OpenAI

client = OpenAI(api_key="YOUR_OPENAI_API_KEY")

class TestBackgroundRecognitionFunctions(unittest.TestCase):

    def test_capture_background_image(self):
        image_path = capture_background_image()
        self.assertIsNotNone(image_path)
        self.assertTrue(image_path.exists())

    def test_analyze_background(self):
        image_path = capture_background_image()
        analysis = analyze_background(image_path, client)
        self.assertIsInstance(analysis, str)
        self.assertGreater(len(analysis), 0)

if __name__ == '__main__':
    unittest.main()

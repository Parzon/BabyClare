import unittest
import numpy as np
from audio import capture_audio, generate_voice_embeddings

class TestAudioFunctions(unittest.TestCase):

    def test_capture_audio(self):
        audio = capture_audio()
        self.assertIsNotNone(audio)
        self.assertEqual(len(audio.shape), 2)  # Stereo audio
        self.assertGreater(audio.shape[0], 0)  # Non-empty audio

    def test_generate_voice_embeddings(self):
        audio = capture_audio()
        embeddings = generate_voice_embeddings(audio)
        self.assertEqual(len(embeddings), 512)  # Assuming embedding size is 512

if __name__ == '__main__':
    unittest.main()

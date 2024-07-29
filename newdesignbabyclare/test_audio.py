import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from audio import listen_to_user, generate_voice_embeddings

class TestAudioFunctions(unittest.TestCase):

    @patch('audio.sd')
    @patch('audio.wv')
    def test_listen_to_user(self, mock_wv, mock_sd):
        mock_sd.rec.return_value = np.zeros((44100 * 7, 2))
        mock_sd.wait.return_value = None
        mock_wv.write.return_value = None
        listen_to_user()
        mock_sd.rec.assert_called_once()
        mock_sd.wait.assert_called_once()
        mock_wv.write.assert_called_once()

    def test_generate_voice_embeddings(self):
        audio = np.random.rand(44100 * 7)
        embeddings = generate_voice_embeddings(audio)
        self.assertEqual(len(embeddings), 512)  # Assuming embedding size is 512

if __name__ == '__main__':
    unittest.main()

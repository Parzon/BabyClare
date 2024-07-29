import unittest
import asyncio
from unittest.mock import patch, MagicMock
from main import main

class TestMainScript(unittest.TestCase):

    @patch('audio.listen_to_user')
    @patch('video.capture_video_frames')
    @patch('audio.analyze_voice_sentiment')
    @patch('video.analyze_face_sentiment')
    def test_main_execution(self, mock_analyze_face_sentiment, mock_analyze_voice_sentiment, mock_capture_video_frames, mock_listen_to_user):
        mock_listen_to_user.return_value = None
        mock_capture_video_frames.return_value = iter([np.zeros((224, 224, 3), dtype=np.uint8)])
        mock_analyze_voice_sentiment.return_value = asyncio.Future()
        mock_analyze_voice_sentiment.return_value.set_result(None)
        mock_analyze_face_sentiment.return_value = asyncio.Future()
        mock_analyze_face_sentiment.return_value.set_result(None)
        
        asyncio.run(main())
        mock_listen_to_user.assert_called_once()
        mock_capture_video_frames.assert_called_once()

if __name__ == '__main__':
    unittest.main()

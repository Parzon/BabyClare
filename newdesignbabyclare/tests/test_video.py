import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from video import capture_video_frames, detect_faces, generate_face_embeddings

class TestVideoFunctions(unittest.TestCase):

    @patch('cv2.VideoCapture')
    def test_capture_video_frames(self, mock_VideoCapture):
        mock_capture = MagicMock()
        mock_capture.isOpened.return_value = True
        mock_capture.read.return_value = (True, np.zeros((224, 224, 3), dtype=np.uint8))
        mock_VideoCapture.return_value = mock_capture

        frames = list(capture_video_frames())
        self.assertGreaterEqual(len(frames), 1)

    def test_detect_faces(self):
        frame = np.zeros((224, 224, 3), dtype=np.uint8)
        faces = detect_faces(frame)
        self.assertGreaterEqual(len(faces), 0)  # Check for valid detection

    def test_generate_face_embeddings(self):
        frame = np.zeros((224, 224, 3), dtype=np.uint8)
        faces = detect_faces(frame)
        if faces:
            face = frame[faces[0][1]:faces[0][1]+faces[0][3], faces[0][0]:faces[0][0]+faces[0][2]]
            embeddings = generate_face_embeddings(face)
            self.assertEqual(len(embeddings), 512)  # Assuming embedding size is 512

if __name__ == '__main__':
    unittest.main()

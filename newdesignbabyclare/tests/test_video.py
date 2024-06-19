# test_video.py

import unittest
import cv2
import numpy as np
from video import capture_video_frames, detect_faces, generate_face_embeddings

class TestVideoFunctions(unittest.TestCase):

    def test_detect_faces(self):
        frame = cv2.imread('path_to_sample_image.jpg')
        faces = detect_faces(frame)
        self.assertGreaterEqual(len(faces), 1)  # At least one face detected

    def test_generate_face_embeddings(self):
        frame = cv2.imread('path_to_sample_image.jpg')
        faces = detect_faces(frame)
        face = frame[faces[0][1]:faces[0][1]+faces[0][3], faces[0][0]:faces[0][0]+faces[0][2]]
        embeddings = generate_face_embeddings(face)
        self.assertEqual(len(embeddings), 512)  # Assuming embedding size is 512

if __name__ == '__main__':
    unittest.main()

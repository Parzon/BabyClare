# test_database.py

import unittest
import sqlite3
import numpy as np
from database import setup_database, add_user, get_user_by_id, add_interaction

class TestDatabaseFunctions(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        setup_database()

    def test_add_user(self):
        face_embedding = np.random.rand(128)
        voice_embedding = np.random.rand(128)
        user_id = add_user("Test User", face_embedding, voice_embedding)
        self.assertGreater(user_id, 0)

    def test_get_user_by_id(self):
        user = get_user_by_id(1)
        self.assertIsNotNone(user)
        self.assertEqual(user[1], "Test User")

    def test_add_interaction(self):
        add_interaction(1, "Test input", "Test response", "Happy", "Neutral")
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("SELECT * FROM interactions WHERE user_id=?", (1,))
        interaction = c.fetchone()
        conn.close()
        self.assertIsNotNone(interaction)
        self.assertEqual(interaction[2], "Test input")

if __name__ == '__main__':
    unittest.main()

import sqlite3
import numpy as np

def setup_database():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY, name TEXT, face_embedding BLOB, voice_embedding BLOB)''')
    c.execute('''CREATE TABLE IF NOT EXISTS interactions
                 (id INTEGER PRIMARY KEY, user_id INTEGER, timestamp TEXT, input_text TEXT, response_text TEXT, 
                  voice_sentiment TEXT, face_sentiment TEXT, FOREIGN KEY(user_id) REFERENCES users(id))''')
    conn.commit()
    conn.close()

def add_user(name, face_embedding, voice_embedding):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("INSERT INTO users (name, face_embedding, voice_embedding) VALUES (?, ?, ?)",
              (name, face_embedding.tobytes(), voice_embedding.tobytes()))
    conn.commit()
    user_id = c.lastrowid
    conn.close()
    return user_id

def get_user_by_id(user_id):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE id=?", (user_id,))
    user = c.fetchone()
    conn.close()
    return user

def add_interaction(user_id, input_text, response_text, voice_sentiment, face_sentiment):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    c.execute("INSERT INTO interactions (user_id, timestamp, input_text, response_text, voice_sentiment, face_sentiment) VALUES (?, ?, ?, ?, ?, ?)",
              (user_id, timestamp, input_text, response_text, voice_sentiment, face_sentiment))
    conn.commit()
    conn.close()

def find_closest_face(face_embedding):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT id, name, face_embedding FROM users")
    users = c.fetchall()
    conn.close()

    closest_user = None
    closest_distance = float('inf')

    for user in users:
        user_id, name, stored_face_embedding = user
        stored_face_embedding = np.frombuffer(stored_face_embedding, dtype=np.float32)
        distance = cosine(face_embedding, stored_face_embedding)
        if distance < closest_distance:
            closest_distance = distance
            closest_user = (name, user_id)

    return closest_user if closest_distance < 0.6 else None  # Placeholder threshold

def find_closest_voice(voice_embedding):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT id, name, voice_embedding FROM users")
    users = c.fetchall()
    conn.close()

    closest_user = None
    closest_distance = float('inf')

    for user in users:
        user_id, name, stored_voice_embedding = user
        stored_voice_embedding = np.frombuffer(stored_voice_embedding, dtype=np.float32)
        distance = cosine(voice_embedding, stored_voice_embedding)
        if distance < closest_distance:
            closest_distance = distance
            closest_user = (name, user_id)

    return closest_user if closest_distance < 0.6 else None  # Placeholder threshold

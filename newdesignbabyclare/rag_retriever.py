# rag_retriever.py

import sqlite3
import openai

def retrieve_interactions(user_id):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT * FROM interactions WHERE user_id=?", (user_id,))
    interactions = c.fetchall()
    conn.close()
    return interactions

def generate_contextual_response(text, sentiment, interactions, client):
    context = "Here are the past interactions with this user:\n"
    for interaction in interactions:
        context += f"User: {interaction[3]}, Bot: {interaction[4]}, Voice Sentiment: {interaction[5]}, Face Sentiment: {interaction[6]}\n"
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a friendly virtual companion."},
            {"role": "user", "content": f"Context: {context}\nUser Input: {text}\nSentiment: {sentiment}"}
        ]
    )
    return response['choices'][0]['message']['content']

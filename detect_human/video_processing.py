import cv2
import face_recognition
import sqlite3
import numpy as np
import uuid
from datetime import datetime

# Инициализация БД
conn = sqlite3.connect('people.db', check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS people (
        id TEXT PRIMARY KEY,
        embedding BLOB,
        name TEXT,
        created_at TIMESTAMP
    )
''')
conn.commit()


def process_frame(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, faces)

    for encoding in encodings:
        encoding_bytes = encoding.tobytes()

        # Поиск существующей записи
        cursor.execute('SELECT id, embedding FROM people')
        match_id = None
        for row in cursor.fetchall():
            stored_id, stored_emb = row
            stored_encoding = np.frombuffer(stored_emb, dtype=np.float64)
            distance = face_recognition.face_distance([stored_encoding], encoding)[0]

            if distance < 0.6:
                match_id = stored_id
                break

        # Добавление нового пользователя
        if not match_id:
            match_id = str(uuid.uuid4())
            cursor.execute('INSERT INTO people (id, embedding, created_at) VALUES (?,?,?)',
                           (match_id, encoding_bytes, datetime.now()))
            conn.commit()

    return faces


# Запуск видеопотока
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret: break

    faces = process_frame(frame)

    # Отрисовка прямоугольников
    for (top, right, bottom, left) in faces:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

    cv2.imshow('Face Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
conn.close()
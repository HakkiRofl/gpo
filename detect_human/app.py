import cv2
import face_recognition
import sqlite3
import numpy as np
import uuid
from datetime import datetime
from flask import Flask, render_template, request, jsonify, Response
import threading
import os

app = Flask(__name__)

# Конфигурация
UPLOAD_FOLDER = 'static/uploads'
DATABASE = 'people.db'
VIDEO_SOURCE = 0
FRAME_SCALE = 1  # Уменьшение размера кадра для оптимизации

# Инициализация системы
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Инициализация базы данных
def init_db():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS people (
            id TEXT PRIMARY KEY,
            embedding BLOB NOT NULL,
            name TEXT NOT NULL DEFAULT 'Unknown',
            created_at TIMESTAMP NOT NULL,
            photo_path TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()


init_db()

# Настройки видеопотока
video_capture = cv2.VideoCapture(VIDEO_SOURCE)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640 * FRAME_SCALE)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480 * FRAME_SCALE)
video_capture.set(cv2.CAP_PROP_FPS, 30)
lock = threading.Lock()


def process_frame(frame):
    small_frame = cv2.resize(frame, (0, 0), fx=FRAME_SCALE, fy=FRAME_SCALE)
    rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, face_locations)
    return face_locations, encodings, small_frame


def save_new_face(face_image, encoding):
    face_id = str(uuid.uuid4())
    filename = f"{face_id}.jpg"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    cv2.imwrite(filepath, face_image)

    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO people (id, embedding, created_at, photo_path)
        VALUES (?,?,?,?)
    ''', (face_id, encoding.tobytes(), datetime.now(), filepath))
    conn.commit()
    conn.close()
    return face_id


def recognize_faces(encodings):
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    recognized = []

    cursor.execute('SELECT id, name, embedding FROM people')
    known_faces = [(row[0], row[1], np.frombuffer(row[2], dtype=np.float64))
                   for row in cursor.fetchall()]

    for encoding in encodings:
        matches = face_recognition.compare_faces(
            [face[2] for face in known_faces],
            encoding,
            tolerance=0.55
        )

        face_ids = [known_faces[i][0] for i, match in enumerate(matches) if match]
        names = [known_faces[i][1] or "Unknown" for i, match in enumerate(matches) if match]

        if face_ids:
            face_id = face_ids[0]
            name = names[0]
        else:
            face_id = None
            name = "Unknown"

        recognized.append({'id': face_id, 'name': name})

    conn.close()
    return recognized


def generate_frames():
    while True:
        with lock:
            success, frame = video_capture.read()
            if not success:
                break

            face_locations, encodings, small_frame = process_frame(frame)
            recognized_faces = recognize_faces(encodings) if encodings else []

            # Сохранение новых лиц
            for i, face_data in enumerate(recognized_faces):
                if face_data['id'] is None and face_locations:
                    top, right, bottom, left = [int(c / FRAME_SCALE) for c in face_locations[i]]
                    face_image = frame[top:bottom, left:right]
                    face_id = save_new_face(face_image, encodings[i])
                    face_data['id'] = face_id
                    face_data['name'] = "Unknown"

            # Отрисовка элементов
            for (top, right, bottom, left), face_data in zip(face_locations, recognized_faces):
                top = int(top / FRAME_SCALE)
                right = int(right / FRAME_SCALE)
                bottom = int(bottom / FRAME_SCALE)
                left = int(left / FRAME_SCALE)

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                label = f"{face_data['name']} ({face_data['id'][:8]})" if face_data['id'] else "New Face"
                cv2.putText(frame, label, (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('SELECT id, name, created_at, photo_path FROM people')
    users = [{"id": row[0], "name": row[1], "date": row[2], "photo": row[3]}
             for row in cursor.fetchall()]
    conn.close()
    return render_template('ui.html', users=users)


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/users', methods=['GET', 'POST', 'DELETE'])
def manage_users():
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()

        if request.method == 'POST':
            # Добавляем подробное логирование и проверку данных
            data = request.get_json()
            print("Received update request:", data)  # Логирование

            if not data or 'id' not in data or 'name' not in data:
                return jsonify({"error": "Invalid request format"}), 400

            user_id = data['id']
            new_name = data['name'].strip()

            if not new_name:
                return jsonify({"error": "Name cannot be empty"}), 400

            # Выполняем обновление
            cursor.execute(
                'UPDATE people SET name = ? WHERE id = ?',
                (new_name, user_id)
            )

            # Проверяем количество изменённых строк
            if cursor.rowcount == 0:
                return jsonify({"error": "User not found"}), 404

            conn.commit()

            # Возвращаем полные данные обновлённого пользователя
            cursor.execute(
                'SELECT id, name, photo_path FROM people WHERE id = ?',
                (user_id,)
            )
            updated_user = cursor.fetchone()

            return jsonify({
                "status": "success",
                "user": {
                    "id": updated_user[0],
                    "name": updated_user[1],
                    "photo": updated_user[2]
                }
            })

        elif request.method == 'DELETE':
            user_id = request.json.get('id')
            if not user_id:
                return jsonify({"error": "Missing user ID"}), 400

            # Получаем путь к файлу
            cursor.execute('SELECT photo_path FROM people WHERE id=?', (user_id,))
            result = cursor.fetchone()
            if result:
                photo_path = result[0]
                if os.path.exists(photo_path):
                    os.remove(photo_path)

            # Удаляем запись из БД
            cursor.execute('DELETE FROM people WHERE id=?', (user_id,))
            conn.commit()
            return jsonify({"status": "success", "deleted_id": user_id})

        else:
            cursor.execute('SELECT id, name, created_at, photo_path FROM people')
            users = [{"id": row[0], "name": row[1], "date": row[2], "photo": row[3]}
                     for row in cursor.fetchall()]
            return jsonify(users)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()


@app.route('/upload', methods=['GET', 'POST'])
def upload_photo():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        name = request.form.get('name', 'Unknown')

        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        try:
            img = face_recognition.load_image_file(file)
            encodings = face_recognition.face_encodings(img)

            if not encodings:
                return jsonify({"error": "No faces detected"}), 400

            face_id = save_new_face(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), encodings[0])
            conn = sqlite3.connect(DATABASE)
            cursor = conn.cursor()
            cursor.execute('UPDATE people SET name=? WHERE id=?', (name, face_id))
            conn.commit()
            conn.close()

            return jsonify({"status": "success", "id": face_id})

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return render_template('upload.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
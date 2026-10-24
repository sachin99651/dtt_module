from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import time
import threading
import base64
import os

# Initialize Flask and SocketIO
app = Flask(__name__, template_folder='.')
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*") 

# ─── Load Models ────────────────────────────────────────────────────────────
print("Loading models...")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Words model (Conv1D / sequence-based - expects 75 frames)
model_words = load_model(os.path.join(BASE_DIR, 'sign_language_model_cnn.h5'))
gestures_words = np.array(['hello', 'thank_you', 'yes', 'no', 'goodbye'])

# Alphabet model (Static Dense Network - expects 1 frame)
model_alpha = load_model(os.path.join(BASE_DIR, 'sign_language_model_alphabet.h5'))
gestures_alpha = np.array([
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'nothing'
])

print("Both models loaded.")

@app.route('/health')
def health():
    return jsonify({
        "status": "ok",
        "models": {
            "alphabet": list(model_alpha.input_shape[1:]),
            "words": list(model_words.input_shape[1:]),
        }
    })

# ─── MediaPipe Setup ─────────────────────────────────────────────────────────
sequence_length = 75 
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils # Added drawing utility
hands_processor = mp_hands.Hands(
    static_image_mode=False, max_num_hands=1,
    min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands_lock = threading.Lock()

# ─── Per-Session State ────────────────────────────────────────────────────────
sessions = {}
sessions_lock = threading.Lock()

def get_session(session_id):
    with sessions_lock:
        if session_id not in sessions:
            sessions[session_id] = {
                "mode": "words",      
                "running": False,
                "word": "...",
                "confidence": 0.0,
                "display_until": 0,
                "sequence": [],       
            }
        return sessions[session_id]

def extract_landmarks(frame_bgr):
    small_frame = cv2.resize(frame_bgr, (320, 240))
    image_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    with hands_lock:
        results = hands_processor.process(image_rgb)
    if results.multi_hand_landmarks:
        lm = results.multi_hand_landmarks[0]
        landmarks_arr = np.array([[p.x, p.y, p.z] for p in lm.landmark]).flatten()
        return landmarks_arr, results.multi_hand_landmarks
    return None, None

# ─── Flask Routes — HTML Pages ───────────────────────────────────────────────
def serve_html(filename):
    path = os.path.join(os.path.dirname(__file__), 'templates', filename)
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

@app.route('/')
def index(): return serve_html('index.html')
@app.route('/category')
def category(): return serve_html('category.html')
@app.route('/types')
def types(): return serve_html('types.html')
@app.route('/detect')
def detect(): return serve_html('detect.html')
@app.route('/alphabets')
def alphabets(): return serve_html('alphabets.html')

# ─── REST API Routes (Session Controls) ──────────────────────────────────────
@app.route('/start', methods=['POST'])
def http_start():
    data = request.get_json(force=True, silent=True) or {}
    session_id = data.get('session_id', 'default')
    mode = data.get('mode', 'words')
    sess = get_session(session_id)
    with sessions_lock:
        sess['mode'] = mode
        sess['running'] = True
        sess['sequence'] = []
        sess['word'] = '...'
        sess['confidence'] = 0.0
        sess['display_until'] = 0
    return jsonify({"status": "started", "mode": mode})

@app.route('/stop', methods=['POST'])
def http_stop():
    data = request.get_json(force=True, silent=True) or {}
    session_id = data.get('session_id', 'default')
    sess = get_session(session_id)
    with sessions_lock:
        sess['running'] = False
        sess['sequence'] = []
        sess['word'] = '...'
        sess['confidence'] = 0.0
    return jsonify({"status": "stopped"})

@app.route('/set_mode', methods=['POST'])
def http_set_mode():
    data = request.get_json(force=True, silent=True) or {}
    session_id = data.get('session_id', 'default')
    mode = data.get('mode', 'words')
    sess = get_session(session_id)
    with sessions_lock:
        sess['mode'] = mode
        sess['sequence'] = []
    return jsonify({"status": "ok", "mode": mode})

# ─── WebSocket Event for Real-Time Frame Processing ──────────────────────────
@socketio.on('process_frame_ws')
def handle_process_frame_ws(data):
    session_id = data.get('session_id', request.sid)
    frame_b64 = data.get('frame', '')
    sess = get_session(session_id)

    if not sess['running']:
        emit('frame_result', {"running": False, "word": None, "confidence": 0})
        return

    try:
        header, encoded = frame_b64.split(',', 1) if ',' in frame_b64 else ('', frame_b64)
        img_bytes = base64.b64decode(encoded)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("imdecode returned None")
    except Exception as e:
        emit('frame_result', {"running": sess['running'], "error": str(e), "word": None, "confidence": 0})
        return

    landmarks, multi_hand_landmarks = extract_landmarks(frame)
    mode = sess['mode']
    now = time.time()

    # ── Alphabet mode (Single-Frame Static) ──
    if mode == 'alphabet':
        confidence_threshold = 0.60
        if landmarks is not None:
            input_data = np.expand_dims(landmarks, axis=0) 
            try:
                prediction = model_alpha(input_data, training=False).numpy()[0]
            except (ValueError, RuntimeError) as error:
                emit('frame_result', {
                    "running": True, "error": f"Alphabet model inference failed: {error}",
                    "word": None, "confidence": 0
                })
                return
            conf = float(np.max(prediction))

            if conf >= confidence_threshold:
                letter = gestures_alpha[np.argmax(prediction)]
                if letter != 'nothing':
                    if letter != sess['word'] or now > sess['display_until']:
                        sess['word'] = letter
                        sess['confidence'] = conf
                        sess['display_until'] = now + 1.0

        if time.time() > sess['display_until']:
            sess['word'] = '...'
            sess['confidence'] = 0.0

    # ── Words mode (75-Frame Sequence) ──
    elif mode == 'words':
        confidence_threshold = 0.80
        if landmarks is not None:
            sess['sequence'].append(landmarks)
        sess['sequence'] = sess['sequence'][-sequence_length:]

        if len(sess['sequence']) == sequence_length:
            input_data = np.expand_dims(sess['sequence'], axis=0)
            try:
                prediction = model_words(input_data, training=False).numpy()[0]
            except (ValueError, RuntimeError) as error:
                emit('frame_result', {
                    "running": True, "error": f"Words model inference failed: {error}",
                    "word": None, "confidence": 0
                })
                return
            conf = float(np.max(prediction))

            if conf >= confidence_threshold:
                word = gestures_words[np.argmax(prediction)]
                if word != sess['word'] or now > sess['display_until']:
                    sess['word'] = word
                    sess['confidence'] = conf
                    sess['display_until'] = now + 2.0

        if time.time() > sess['display_until']:
            sess['word'] = '...'
            sess['confidence'] = 0.0

    # Draw landmarks on the frame if they exist
    if multi_hand_landmarks:
        for hand_landmarks in multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS)
                
    # Encode the frame with drawn landmarks to Base64 to send to frontend
    _, buffer = cv2.imencode('.jpg', frame)
    annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')

    word_out = sess['word'] if sess['word'] != '...' else None
    
    emit('frame_result', {
        "running": True,
        "word": word_out,
        "confidence": round(sess['confidence'] * 100, 1),
        "image": annotated_b64
    })

# ─── Run ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n* Starting WebSocket Server at http://0.0.0.0:5001")
    print("  Open http://localhost:5001 in your browser\n")
    socketio.run(app, debug=False, host='0.0.0.0', port=5001)
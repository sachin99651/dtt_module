import cv2
import os
import numpy as np
import mediapipe as mp

# --- Constants ---
PROCESSED_DATA_PATH = "processed_data"
gestures = np.array(['hello', 'thank_you', 'yes', 'no', 'goodbye'])
num_sequences = 75
sequence_length = 75

# --- Initialize MediaPipe ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, max_num_hands=1, 
    min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# --- Create Folder ---
if not os.path.exists(PROCESSED_DATA_PATH):
    os.makedirs(PROCESSED_DATA_PATH)

# --- Main Loop ---
cap = cv2.VideoCapture(1)
sequences, labels = [], []
label_map = {label: num for num, label in enumerate(gestures)}

for gesture in gestures:
    for sequence in range(num_sequences):
        window = []
        for frame_num in range(sequence_length):
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if frame_num == 0:
                cv2.putText(frame, 'STARTING COLLECTION', (120, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                cv2.putText(frame, f'Collecting frames for {gesture} | Video {sequence}', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.imshow('OpenCV Feed', frame)
                cv2.waitKey(2000)
            else:
                cv2.putText(frame, f'Collecting frames for {gesture} | Video {sequence}', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.imshow('OpenCV Feed', frame)
            
            res = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten() if results.multi_hand_landmarks else np.zeros(21 * 3)
            window.append(res)
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                exit()
        
        sequences.append(window)
        labels.append(label_map[gesture])

cap.release()
cv2.destroyAllWindows()

X = np.array(sequences)
y = np.array(labels)
np.save(os.path.join(PROCESSED_DATA_PATH, 'X_data.npy'), X)
np.save(os.path.join(PROCESSED_DATA_PATH, 'y_labels.npy'), y)

"""
PASSENGER EMOTION & FATIGUE DETECTION MODULE

Core Functions:
- Real-time emotion detection (happy, sad, angry, neutral, fear, surprised)
- Drowsiness monitoring via eye aspect ratio (EAR) tracking
- Yawning detection using mouth aspect ratio (MAR)
- Head nodding detection for fatigue assessment
- Multi-face support (up to 5 faces simultaneously)

Key Features:
- Custom emotion correction rules to reduce false positives
- Temporal filtering using emotion history buffers
- Enhanced image processing integration via smart_enhance()
- Priority-based state reporting (Drowsiness > Yawning > Nodding > Emotion)

Usage: Main detection engine for passenger monitoring in automotive applications.
Can run standalone or be imported by data collection/testing scripts.
"""

import cv2
import mediapipe as mp
from deepface import DeepFace
import numpy as np
from collections import deque, Counter, defaultdict
from compare_crop_enhance import smart_enhance

# =========== Parameters ===========
EAR_THRESHOLD = 0.25
EAR_CONSEC_FRAMES = 15
MAR_THRESHOLD = 0.7
NOD_THRESHOLD = 10

# =========== Angry probability accumulator ===========
ANGRY_SMOOTH_WINDOW = 10   # number of frames to average over
ANGRY_TEMPERATURE   = 0.6  # sharpens DeepFace probabilities (< 1.0 = sharper)
ANGRY_MIN_CONF      = 0.30 # minimum averaged probability to confirm angry

# =========== Indices for MediaPipe Landmarks ===========
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
NOSE_TIP_IDX = 1

# DeepFace emotion order (must match what DeepFace returns)
DEEPFACE_EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# =========== State (per face) ===========
blink_counter = defaultdict(int)
yawn_counter = defaultdict(int)
nod_counter = defaultdict(int)
prev_y = defaultdict(lambda: None)
emotion_history = defaultdict(lambda: deque(maxlen=15))
neutral_like_sad = defaultdict(int)

# Angry probability history per face
angry_prob_history = defaultdict(lambda: deque(maxlen=ANGRY_SMOOTH_WINDOW))

# =========== Helper Functions ===========
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio_mediapipe(landmarks):
    top = landmarks[13]
    bottom = landmarks[14]
    left = landmarks[78]
    right = landmarks[308]
    mar = np.linalg.norm(top - bottom) / np.linalg.norm(left - right)
    return mar

def angry_softmax_temperature(probs_dict, temp=ANGRY_TEMPERATURE):
    """
    Apply temperature scaling to DeepFace's probability dict.
    Sharpens the distribution when temp < 1.0 so confident
    predictions get amplified and weak noise gets suppressed.
    Returns a numpy array in DEEPFACE_EMOTIONS order.
    """
    vec = np.array([probs_dict.get(e, 0.0) for e in DEEPFACE_EMOTIONS], dtype=np.float64)
    vec = np.clip(vec, 1e-9, None)
    log_scaled = np.log(vec) / temp
    log_scaled -= log_scaled.max()
    scaled = np.exp(log_scaled)
    return scaled / scaled.sum()

def is_angry_by_probability(face_idx, raw_probs_dict):
    """
    Accumulates temperature-scaled probability vectors across frames
    and returns True if the mean angry probability exceeds ANGRY_MIN_CONF.
    This replaces all geometry-based angry heuristics.
    """
    prob_vec = angry_softmax_temperature(raw_probs_dict)
    angry_prob_history[face_idx].append(prob_vec)
    mean_vec = np.mean(angry_prob_history[face_idx], axis=0)
    angry_idx = DEEPFACE_EMOTIONS.index('angry')
    return float(mean_vec[angry_idx]) >= ANGRY_MIN_CONF

def get_face_mesh():
    mp_face_mesh = mp.solutions.face_mesh
    return mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=5, refine_landmarks=True)

def analyze_faces_and_draw(frame, face_mesh=None):
    """
    Enhance the image, detect faces, emotions, drowsiness, yawning, nodding, and draw overlays.
    Returns the processed frame and a list of detected states (one per face):
    'Drowsiness', 'Yawning', 'Head Nodding', or the current emotion.
    """
    if face_mesh is None:
        face_mesh = get_face_mesh()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    h, w, _ = frame.shape
    states = []
    if results.multi_face_landmarks:
        for face_idx, face_landmarks in enumerate(results.multi_face_landmarks):
            landmarks = np.array([(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark])
            nose_point = landmarks[NOSE_TIP_IDX]
            xs, ys = landmarks[:, 0], landmarks[:, 1]
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()
            margin = 40
            x_min_c = max(x_min - margin, 0)
            x_max_c = min(x_max + margin, w)
            y_min_c = max(y_min - margin, 0)
            y_max_c = min(y_max + margin, h)
            face_img = frame[y_min_c:y_max_c, x_min_c:x_max_c]
            try:
                result = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False, detector_backend='mediapipe')
                emotion = result[0]['dominant_emotion']
                raw_probs = result[0]['emotion']   # full probability dict
            except Exception:
                emotion = "Unknown"
                raw_probs = {}
            leftEye = landmarks[LEFT_EYE_IDX]
            rightEye = landmarks[RIGHT_EYE_IDX]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            mar = mouth_aspect_ratio_mediapipe(landmarks)
            # Custom rule: If 'fear' but mouth and eyes are not wide open, treat as neutral
            if emotion == "fear":
                if mar < 0.5 and ear < 0.3:
                    emotion = "neutral"
            # =========== ANGRY DETECTION (probability accumulator) ===========
            # Accumulate DeepFace's angry probability across frames.
            # Confirm angry only when the running average is strong enough.
            # This replaces all the old geometry brow/eye/jaw rules for angry.
            if raw_probs:
                if is_angry_by_probability(face_idx, raw_probs):
                    emotion = "angry"
                elif emotion == "angry":
                    # DeepFace said angry this frame but accumulator not yet
                    # confident enough — hold off to avoid false positives
                    emotion = "neutral"
            # =========== END ANGRY DETECTION ===========
            # Custom rule: If 'surprised' but eyes/mouth not wide open, treat as neutral
            if emotion == "surprised":
                if ear < 0.22 or mar < 0.32:
                    emotion = "neutral"
            # Force 'surprised' if geometry is strongly surprised, even if DeepFace says 'fear'
            if (ear > 0.28 and mar > 0.5) and emotion in ["fear", "neutral"]:
                emotion = "surprised"
            # Drowsiness detection: only if eyes closed for >2.5 seconds AND head is tilted
            drowsy = False
            EYES_CLOSED_FRAMES = 50  # 2.5 seconds at 20 FPS
            eyes_closed_long = blink_counter[face_idx] >= EYES_CLOSED_FRAMES
            head_center_y = (y_min + y_max) // 2
            head_center_x = (x_min + x_max) // 2
            head_tilted = abs(nose_point[0] - head_center_x) > 10 and nose_point[1] > head_center_y + 5
            if eyes_closed_long and head_tilted:
                drowsy = True
                cv2.putText(frame, "⚠ Drowsiness Detected!", (x_min, y_max + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if ear < EAR_THRESHOLD:
                blink_counter[face_idx] += 1
            else:
                blink_counter[face_idx] = 0
            # Yawning detection
            yawning = False
            if mar > MAR_THRESHOLD:
                yawn_counter[face_idx] += 1
                if yawn_counter[face_idx] > 0:
                    yawning = True
            else:
                yawn_counter[face_idx] = 0
            # Head nodding detection
            nodding = False
            if prev_y[face_idx] is not None and abs(prev_y[face_idx] - nose_point[1]) > NOD_THRESHOLD:
                nod_counter[face_idx] += 1
                if nod_counter[face_idx] > 0:
                    nodding = True
            prev_y[face_idx] = nose_point[1]
            if mar > MAR_THRESHOLD:
                yawn_counter[face_idx] += 1
               # cv2.putText(frame, "⚠ Yawning!", (x_min, y_max + 70),
                           # cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            if emotion == "sad":
                if mar < 0.3 and ear > EAR_THRESHOLD:
                    neutral_like_sad[face_idx] += 1
                else:
                    neutral_like_sad[face_idx] = 0
                if neutral_like_sad[face_idx] >= 5:
                    emotion = "neutral"
            left_corner_y = landmarks[78][1]
            right_corner_y = landmarks[308][1]
            center_top_y = landmarks[13][1]
            center_bottom_y = landmarks[14][1]
            center_y = (center_top_y + center_bottom_y) / 2
            if left_corner_y > center_y and right_corner_y > center_y:
                emotion = "sad"
            emotion_history[face_idx].append(emotion)
            fear_count = sum([e == "fear" for e in emotion_history[face_idx]])
            if fear_count < 0.4 * len(emotion_history[face_idx]):
                most_common_emotion = Counter(emotion_history[face_idx]).most_common(1)[0][0]
                display_emotion = most_common_emotion
            else:
                display_emotion = "fear"
            # Draw overlays
            color = (0, 0, 255) if display_emotion == "angry" else (0, 255, 0)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.putText(frame, f"Emotion: {display_emotion}", (x_min, y_min-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            for (x, y) in landmarks:
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
            # Priority: Drowsiness > Yawning > Head Nodding > Emotion
            if drowsy:
                state = "Drowsiness"
            elif yawning:
                state = "Yawning"
            elif nodding:
                state = "Head Nodding"
            else:
                state = display_emotion
            states.append(state)
    return frame, states

# =========== Standalone Demo ===========
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print("Passenger camera monitoring started...")
    face_mesh = get_face_mesh()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame, emotion = analyze_faces_and_draw(frame, face_mesh)
        cv2.imshow("Passenger Emotion & Fatigue Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
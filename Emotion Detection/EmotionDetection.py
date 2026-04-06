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

# =========== Indices for MediaPipe Landmarks ===========
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
NOSE_TIP_IDX = 1

# =========== State (per face) ===========
blink_counter = defaultdict(int)
yawn_counter = defaultdict(int)
nod_counter = defaultdict(int)
prev_y = defaultdict(lambda: None)
emotion_history = defaultdict(lambda: deque(maxlen=15))
neutral_like_sad = defaultdict(int)

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
            nose_point = landmarks[NOSE_TIP_IDX]  # <-- Move this up before drowsiness logic
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
            except Exception:
                emotion = "Unknown"
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
            # Custom rule: If 'angry' but eyebrows not lowered or eyes not squinted, treat as neutral
            if emotion == "angry":
                left_brow = landmarks[70][1]
                left_eye_top = landmarks[159][1]
                right_brow = landmarks[300][1]
                right_eye_top = landmarks[386][1]
                # Loosened: allow even larger brow-eye distance and higher EAR for easier detection
                if (left_brow - left_eye_top > 45) and (right_brow - right_eye_top > 45):
                    emotion = "neutral"
                if ear > 0.32:
                    emotion = "neutral"
            # Custom rule: If 'surprised' but eyes/mouth not wide open, treat as neutral
            if emotion == "surprised":
                # Further relax thresholds for EAR and MAR
                if ear < 0.22 or mar < 0.32:
                    emotion = "neutral"
            # Force 'surprised' if geometry is strongly surprised, even if DeepFace says 'fear'
            if (ear > 0.28 and mar > 0.5) and emotion in ["fear", "neutral"]:
                emotion = "surprised"
            # Custom rule: Detect 'angry' if eyebrows are close together, forehead wrinkled, and mouth is neutral or slightly sad
            # Eyebrow distance: landmarks 70 (left brow), 300 (right brow)
            # Forehead wrinkle: vertical distance between brow and forehead (landmark 10)
            # Mouth: neutral or slightly sad (corners not above center)
            brow_distance = abs(landmarks[70][0] - landmarks[300][0])
            left_brow_to_forehead = abs(landmarks[70][1] - landmarks[10][1])
            right_brow_to_forehead = abs(landmarks[300][1] - landmarks[10][1])
            left_corner_y = landmarks[78][1]
            right_corner_y = landmarks[308][1]
            center_top_y = landmarks[13][1]
            center_bottom_y = landmarks[14][1]
            center_y = (center_top_y + center_bottom_y) / 2
            # Typical values: brow_distance < 55 (close), brow_to_forehead < 30 (wrinkled), mouth corners not above center
            if brow_distance < 55 and left_brow_to_forehead < 30 and right_brow_to_forehead < 30:
                if left_corner_y >= center_y and right_corner_y >= center_y:
                    emotion = "angry"
            # Loosened advanced custom rule: Detect 'angry' with more permissive thresholds
            brow_distance = abs(landmarks[70][0] - landmarks[300][0])
            left_brow_y = landmarks[70][1]
            right_brow_y = landmarks[300][1]
            left_brow_inner_y = landmarks[105][1]
            right_brow_inner_y = landmarks[334][1]
            brow_inner_distance = abs(landmarks[105][0] - landmarks[334][0])
            left_upper_lid = landmarks[159][1]
            right_upper_lid = landmarks[386][1]
            left_eye_center = (landmarks[33][1] + landmarks[133][1]) / 2
            right_eye_center = (landmarks[362][1] + landmarks[263][1]) / 2
            nostril_distance = abs(landmarks[97][0] - landmarks[326][0])
            nose_bridge = landmarks[6][1]
            left_nostril = landmarks[97][1]
            right_nostril = landmarks[326][1]
            nose_wrinkle = min(abs(nose_bridge - left_nostril), abs(nose_bridge - right_nostril))
            mouth_height = abs(landmarks[13][1] - landmarks[14][1])
            mouth_width = abs(landmarks[78][0] - landmarks[308][0])
            left_corner_y = landmarks[78][1]
            right_corner_y = landmarks[308][1]
            center_top_y = landmarks[13][1]
            center_bottom_y = landmarks[14][1]
            center_y = (center_top_y + center_bottom_y) / 2
            chin_y = landmarks[152][1]
            jaw_clenched = chin_y < (center_bottom_y + 40)  # Loosened
            # Loosened thresholds for anger cues
            if (
                brow_distance < 70 and  # was 55
                left_brow_y > left_brow_inner_y - 5 and right_brow_y > right_brow_inner_y - 5 and  # allow small difference
                brow_inner_distance < 40 and  # was 30
                ear < 0.28 and  # was 0.23
                left_upper_lid < left_eye_center + 8 and right_upper_lid < right_eye_center + 8 and  # allow higher lids
                nostril_distance > 28 and  # was 35
                nose_wrinkle < 28 and  # was 18
                (mouth_height < 28 or mouth_height > 30) and  # was 18/35
                (left_corner_y >= center_y - 10 or right_corner_y >= center_y - 10) and  # allow corners slightly above
                jaw_clenched
            ):
                emotion = "angry"
            # Restrictive rule: Only set 'angry' if at least four cues are present and DeepFace did not detect a strong emotion
            strong_emotions = ["happy", "surprised", "fear", "disgust", "sad"]
            cues = 0
            if brow_distance < 57:
                cues += 1
            if brow_inner_distance < 31:
                cues += 1
            if ear < 0.26:
                cues += 1
            if mouth_height < 25:
                cues += 1
            if left_corner_y >= center_y - 8 or right_corner_y >= center_y - 8:
                cues += 1
            # Only set angry if cues >= 4 and DeepFace is not confident in another emotion
            if cues >= 4 and emotion not in strong_emotions and (emotion == "neutral" or emotion == "Unknown"):
                emotion = "angry"
            # Drowsiness detection: only if eyes closed for >2.5 seconds AND head is tilted
            drowsy = False
            # Calculate how many frames is 2.5 seconds (assuming 20 FPS)
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
            # if ear < EAR_THRESHOLD:
            #     blink_counter[face_idx] += 1
            #     if blink_counter[face_idx] >= EAR_CONSEC_FRAMES:
            #         cv2.putText(frame, "⚠ Drowsiness Detected!", (x_min, y_max + 50),
            #                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # else:
            #     blink_counter[face_idx] = 0
            if mar > MAR_THRESHOLD:
                yawn_counter[face_idx] += 1
                cv2.putText(frame, "⚠ Yawning!", (x_min, y_max + 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
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
            # At the end of the per-face loop, append display_emotion to emotions
            # emotions.append(display_emotion)
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

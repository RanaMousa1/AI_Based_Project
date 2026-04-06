"""
IMAGE ENHANCEMENT & FACE PROCESSING CORE MODULE

FUNCTIONALITY:
This module provides the foundational image processing pipeline for the emotion detection system,
focusing on intelligent face region extraction and adaptive image enhancement for improved
detection accuracy in varying lighting conditions.

CORE FUNCTIONS:
1. find_faces_and_context(): 
   - Detects all faces in an image using MediaPipe Face Detection
   - Creates intelligent crop region around faces with configurable padding
   - Maintains original aspect ratio to prevent distortion
   - Returns cropped context image, detection results, and offset coordinates

2. smart_enhance():
   - Advanced lighting enhancement using histogram equalization + CLAHE
   - Configurable intensity blending (0.0 = original, 1.0 = fully enhanced)
   - Two-stage enhancement: YUV histogram equalization + LAB CLAHE
   - Designed to improve face detection accuracy in poor lighting

3. detect_emotions_on_context():
   - Performs emotion detection on processed face regions
   - Handles coordinate offset mapping from cropped to original image
   - Uses DeepFace with MediaPipe backend for emotion analysis
   - Returns emotion results with spatial coordinates

SYSTEM INTEGRATION:
- IMPORTED BY: face_detection_passenger.py (smart_enhance)
- IMPORTED BY: side_by_side_with_log.py (smart_enhance) 
- IMPORTED BY: run_compare_on_camera.py (all functions)

USAGE PATTERNS:
1. Real-time Enhancement: smart_enhance() used by passenger detection system
2. Data Collection: smart_enhance() used in logging/recording scripts
3. Testing Pipeline: All functions used in camera comparison testing
4. Standalone Analysis: Can process static images for comparison studies

RESEARCH PURPOSE:
This module represents the core image preprocessing research contribution, demonstrating
that intelligent cropping + adaptive enhancement significantly improves emotion detection
accuracy compared to processing full frames. The comparison functionality allows for
quantitative evaluation of enhancement effectiveness.

STANDALONE MODE:
When run directly, processes "bad-lighting.jpg" and outputs side-by-side comparison
showing: Original → Enhanced → Emotion-Annotated results.
"""

import cv2
import numpy as np
import sys
import mediapipe as mp
from deepface import DeepFace

def find_faces_and_context(img, padding_ratio=0.5):
    mp_face_detection = mp.solutions.face_detection
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if results.detections:
            ih, iw, _ = img.shape
            boxes = []
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                x = int(bboxC.xmin * iw)
                y = int(bboxC.ymin * ih)
                w = int(bboxC.width * iw)
                h = int(bboxC.height * ih)
                boxes.append([x, y, x + w, y + h])
            x_min = min([b[0] for b in boxes])
            y_min = min([b[1] for b in boxes])
            x_max = max([b[2] for b in boxes])
            y_max = max([b[3] for b in boxes])
            # Add padding
            w_union = x_max - x_min
            h_union = y_max - y_min
            x_pad = int(w_union * padding_ratio)
            y_pad = int(h_union * padding_ratio)
            x_min = max(0, x_min - x_pad)
            y_min = max(0, y_min - y_pad)
            x_max = min(iw, x_max + x_pad)
            y_max = min(ih, y_max + y_pad)
            # Maintain original aspect ratio
            crop_w = x_max - x_min
            crop_h = y_max - y_min
            orig_aspect = iw / ih
            crop_aspect = crop_w / crop_h
            if crop_aspect > orig_aspect:
                # Too wide, pad height
                new_h = int(crop_w / orig_aspect)
                pad = (new_h - crop_h) // 2
                y_min = max(0, y_min - pad)
                y_max = min(ih, y_max + pad)
            elif crop_aspect < orig_aspect:
                # Too tall, pad width
                new_w = int(crop_h * orig_aspect)
                pad = (new_w - crop_w) // 2
                x_min = max(0, x_min - pad)
                x_max = min(iw, x_max + pad)
            context_img = img[y_min:y_max, x_min:x_max]
            return context_img, results.detections, (x_min, y_min)
    return img, [], (0, 0)

def smart_enhance(img, intensity=0.5):
    """
    Enhance image lighting using histogram equalization and CLAHE.
    Intensity controls the blend between original and enhanced channels (0.0 = original, 1.0 = fully enhanced).
    """
    # Clamp intensity between 0 and 1
    intensity = max(0.0, min(1.0, intensity))
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    y_eq = cv2.equalizeHist(yuv[:, :, 0])
    # Blend original and equalized Y channel by intensity
    yuv[:, :, 0] = cv2.addWeighted(yuv[:, :, 0], 1.0 - intensity, y_eq, intensity, 0)
    enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))  # Reduce clipLimit for milder effect
    l_clahe = clahe.apply(lab[:, :, 0])
    # Blend original and CLAHE L channel by intensity
    lab[:, :, 0] = cv2.addWeighted(lab[:, :, 0], 1.0 - intensity, l_clahe, intensity, 0)
    final = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return final

def detect_emotions_on_context(processed_img, detections, offset):
    emotions = []
    ih, iw, _ = processed_img.shape
    x_off, y_off = offset
    for detection in detections:
        bboxC = detection.location_data.relative_bounding_box
        # Adjust coordinates for context crop
        x = int(bboxC.xmin * iw)
        y = int(bboxC.ymin * ih)
        w = int(bboxC.width * iw)
        h = int(bboxC.height * ih)
        x, y = max(0, x), max(0, y)
        x2, y2 = min(iw, x + w), min(ih, y + h)
        face_img = processed_img[y:y2, x:x2]
        try:
            result = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False, detector_backend='mediapipe')
            emotions.append((x + x_off, y + y_off, w, h, result[0]['dominant_emotion']))
        except Exception:
            emotions.append((x + x_off, y + y_off, w, h, "Unknown"))
    return emotions

if __name__ == "__main__":
    input_path = "bad-lighting.jpg"  # Default input image
    output_path = "comparison_output.jpg"  # Default output image

    img = cv2.imread(input_path)
    if img is None:
        print("Could not read input image.")
        sys.exit(1)

    # a. Find region of interest (all faces)
    context_img, _, _ = find_faces_and_context(img, padding_ratio=0.5)
    # b/c. Smart image processing on context window
    processed = smart_enhance(context_img)
    # d. Detect faces and emotions on the processed image as if it's a new image
    detections = find_faces_and_context(processed, padding_ratio=0)[1]
    emotions = detect_emotions_on_context(processed, detections, (0, 0))

    # Draw results on the processed image
    annotated = processed.copy()
    for (x, y, w, h, emotion) in emotions:
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(annotated, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Output side-by-side: original, processed, annotated
    h0 = img.shape[0]
    processed_resized = cv2.resize(processed, (int(processed.shape[1] * h0 / processed.shape[0]), h0))
    annotated_resized = cv2.resize(annotated, (processed_resized.shape[1], h0))
    comparison = np.hstack((img, processed_resized, annotated_resized))
    cv2.imwrite(output_path, comparison)
    print(f"Comparison image saved to {output_path}")

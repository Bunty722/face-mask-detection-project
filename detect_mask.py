import cv2
import numpy as np
import tensorflow as tf
import winsound  # Import the built-in Windows sound library

# --- Define the Model Architecture ---
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# --- Load the saved weights ---
model.load_weights('face_mask_detector.h5')

# --- Main detection logic ---
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
labels_dict = {0: 'MASK', 1: 'NO MASK'}
color_dict = {0: (0, 255, 0), 1: (0, 0, 255)}
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        
        if face_roi.size == 0:
            continue

        rgb_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        resized_face = cv2.resize(rgb_face, (150, 150))
        normalized_face = resized_face / 255.0
        reshaped_face = np.reshape(normalized_face, (1, 150, 150, 3))
        
        result = model.predict(reshaped_face)
        label_id = 0 if result[0][0] < 0.5 else 1
        
        # *** USING WINSOUND FOR THE ALERT ***
        if label_id == 1:
            # Beep at 440 Hz for 500 milliseconds (half a second)
            winsound.Beep(440, 500)

        confidence_text = f"{labels_dict[label_id]}: {result[0][0]:.2f}"
        cv2.rectangle(frame, (x, y), (x+w, y+h), color_dict[label_id], 2)
        cv2.putText(frame, confidence_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_dict[label_id], 2)

    cv2.imshow('Face Mask Detector', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
import face_recognition
import numpy as np
import cv2
import os

def load_known_faces_from_folder(folder_path):
    known_encodings = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg', '.JPG')):
            path = os.path.join(folder_path, filename)
            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)
            
            if encodings:
                encoding = encodings[0]
                known_encodings.append(encoding)
            else:
                print(f"No faces detected in {filename}.")
    
    known_names = ["Thaher"] * len(known_encodings)
    return known_encodings, known_names

# Data of me
folder_path = "data" 
known_encodings, known_names = load_known_faces_from_folder(folder_path)

video_capture = cv2.VideoCapture(1) 

video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    # Capture framebyframe
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert the image from (OpenCV format) to (face_recognition format)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Find face location and face encoding in the current frame
    face_locations = face_recognition.face_locations(rgb_frame, model='hog')
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # Compare the face encodings
        distances = face_recognition.face_distance(known_encodings, face_encoding)
        if len(distances) > 0:
            best_match_index = np.argmin(distances)
            
            if distances[best_match_index] < 0.6:
                name = known_names[best_match_index]
                confidence = (1 - distances[best_match_index]) * 100
            else:
                name = "Unknown"
                confidence = 0
            
            face_names.append((name, confidence))
        else:
            face_names.append(("Unknown", 0))

    # Draw results for the current frame
    for (top, right, bottom, left), (name, confidence) in zip(face_locations, face_names):
        if name == "Thaher":
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            label = f"{name} ({confidence:.2f}%)"
            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        elif name == "Unknown":
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, "Unknown", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow('Face Scan System', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

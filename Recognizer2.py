import cv2
import numpy as np
import pyttsx3
import mediapipe as mp

face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
landmark_color = (0, 255, 0)
eye_color = (0, 255, 255)

def load_user_database():
    user_database = {}
    with open("datatext.txt", "r") as file:
        for line in file:
            user_id, _ = line.strip().split(" ")  # Only extract the user ID
            user_database[user_id] = None  # Store user ID without name
    return user_database

def recognize_face(user_database, user_id):
    engine = pyttsx3.init()
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("recognizer/TraningData.yml")

    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            id_, conf = recognizer.predict(roi_gray)
            if conf < 75:
                if str(id_) == user_id:
                    cv2.putText(frame, "Access Granted", (x, y - 10), font, 1, (0, 255, 0), 2)
                    engine.say("Access Granted")
                else:
                    cv2.putText(frame, "Access Denied", (x, y - 10), font, 1, (0, 0, 255), 2)
                    engine.say("Access Denied")
            else:
                cv2.putText(frame, "Face not recognized", (x, y - 10), font, 1, (0, 0, 255), 2)
                engine.say("Face not recognized")

            # Detect eye landmarks and check if the person is looking away
            results = face_mesh.process(frame)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    if detect_looking_away(face_landmarks, frame.shape):
                        print("Looked away")

        cv2.imshow('Face Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def detect_looking_away(face_landmarks, frame_shape):
    left_eye_y = face_landmarks.landmark[159].y * frame_shape[0]
    right_eye_y = face_landmarks.landmark[145].y * frame_shape[0]
    eye_distance_threshold = 10  # Adjust as needed

    if abs(left_eye_y - right_eye_y) > eye_distance_threshold:
        return True  # Person is looking away
    else:
        return False  # Person is not looking away

if __name__ == "__main__":
    user_database = load_user_database()
    user_id = input("Enter your ID: ")

    if user_id in user_database:
        recognize_face(user_database, user_id)
    else:
        print("User ID not found in the database.")

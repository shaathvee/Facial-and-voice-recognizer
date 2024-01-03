import cv2
import subprocess
import streamlit as st
import os
import sounddevice as sd
import soundfile as sf
from PIL import Image

load = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def create_faces(name, user_id):
    f = open("datatext.txt", "a")
    f.write(str(user_id) + " " + name + "\n")
    f.close()

    cap = cv2.VideoCapture(0)
    val = 0

    st.header("Camera Feed")  # Add a header to indicate the camera feed
    img_placeholder = st.empty()

    # Display warning message before recording audio
    st.warning("Read this line: The quick brown fox jumps over the lazy dog")

    # Audio recording
    fs = 44100  # Sample rate
    seconds = 10  # Duration of recording

    audio_placeholder = st.empty()
    audio_data = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    sd.wait()

    # Save audio
    audio_path = f"Data/{user_id}.wav"
    sf.write(audio_path, audio_data, fs)

    # Playback the recorded audio
    sd.play(audio_data, fs)
    sd.wait()

    for i in range(seconds):
        audio_placeholder.write(f"Recording audio... {seconds - i} seconds left.")

    audio_placeholder.success("Audio recorded successfully.")

    while True:
        status, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = load.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            val += 1
            cv2.imwrite(f"Data/{user_id}.{val}.jpg", gray[y:y + h, x:x + w])
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Display the updated image in the Streamlit app
        img_placeholder.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_column_width=True)

        if val >= 50:
            break

    cap.release()
    cv2.destroyAllWindows()
    subprocess.run(['python', 'trainer.py'])

def main():
    st.title("Face Creator")
    name = st.text_input("Enter Username")
    user_id = st.text_input("Enter Unique ID (maxLen = 4)")

    if st.button("Create"):
        create_faces(name, user_id)
        st.success("Data created successfully")

if __name__ == '__main__':
    main()
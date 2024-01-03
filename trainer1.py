import os
import cv2
import numpy as np
from PIL import Image
import librosa

recognizer = cv2.face.LBPHFaceRecognizer_create()
path = 'Data'

def img(path):
    imgpaths = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.endswith('.jpg')]  # getting image paths
    faces = []
    users = []
    for imgpath in imgpaths:
        filename = os.path.basename(imgpath)
        try:
            user = int(filename.split('.')[0])  # getting user ID from filename
        except ValueError:
            print("Invalid filename format:", filename)
            continue  # skip this image
        faceimg = Image.open(imgpath).convert('L')  # converting to grayscale
        facenp = np.array(faceimg, 'uint8')
        faces.append(facenp)
        users.append(user)
        cv2.waitKey(10)

    return users, faces

def audio(path):
    audiopaths = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.endswith('.wav')]  # getting audio paths
    audios = []
    users = []
    for audiopath in audiopaths:
        filename = os.path.basename(audiopath)
        try:
            user = int(filename.split('.')[0])  # getting user ID from filename
        except ValueError:
            print("Invalid filename format:", filename)
            continue  # skip this audio
        audio, _ = librosa.load(audiopath, sr=None)  # load audio file
        audios.append(audio)
        users.append(user)

    return users, audios

users_img, faces = img(path)
users_audio, audios = audio(path)

if len(users_img) != len(users_audio):
    print("Number of images and audio files do not match!")
else:
    users = users_img  # You can choose to use users from images or audios
    recognizer.train(faces, np.array(users))  # training with image data
    recognizer.save('recognizer/TraningData.yml')  # saving trained data for face recognition
    print("Face recognition model trained successfully.")

    # Additional processing for audio files can be added here if needed

cv2.destroyAllWindows()

import cv2
import numpy as np
import sounddevice as sd
import librosa
import tensorflow as tf
from keras.models import load_model
import threading

tf.get_logger().setLevel('ERROR')

face_model = load_model("face_emotion_model.h5")
audio_model = load_model("audio_scream_model.h5")

face_classes = ["neutral","happy","sad","angry","fear","pain"]
audio_classes = ["normal","scream"]

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

latest_sound = None

def record_audio(duration=2, fs=16000):
    global latest_sound
    while True:
        audio = sd.rec(int(duration*fs), samplerate=fs, channels=1)
        sd.wait()
        mfcc = librosa.feature.mfcc(y=np.squeeze(audio), sr=fs, n_mfcc=40)
        mfcc = np.expand_dims(np.mean(mfcc.T, axis=0), axis=(0,-1))
        pred_audio = audio_model.predict(mfcc)
        latest_sound = audio_classes[np.argmax(pred_audio)]

audio_thread = threading.Thread(target=record_audio, daemon=True)
audio_thread.start()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        face = cv2.resize(gray[y:y+h, x:x+w], (48,48)) / 255.0
        face = np.expand_dims(face, axis=(0,-1))
        pred = face_model.predict(face)
        emotion = face_classes[np.argmax(pred)]
        if emotion in ["fear","pain"]:
            print("Help: Distress detected from face")
    if latest_sound == "scream":
        print("Help: Distress detected from voice")
        latest_sound = None
    cv2.imshow("Elder Monitor", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

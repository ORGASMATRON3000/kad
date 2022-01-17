import face_recognition
import cv2
import numpy as np
import os
import glob
from tqdm import tqdm
import time
import transliterate
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing.image import img_to_array
import pickle
emotion_model_path = 'models/_mini_XCEPTION.91-0.64.hdf5'
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised","neutral"]
start = time.time()
faces_encodings = []
faces_names = []
cur_direc = os.getcwd()
path = os.path.join(cur_direc, 'face/')
list_of_files = [f for f in glob.glob(path+'*.jpg')]
number_files = len(list_of_files)
names = list_of_files.copy()
for i in range(number_files):
    globals()['image_{}'.format(i)] = face_recognition.load_image_file(list_of_files[i])
    globals()['image_encoding_{}'.format(i)] = face_recognition.face_encodings(globals()['image_{}'.format(i)])[0]
    faces_encodings.append(globals()['image_encoding_{}'.format(i)])
    names[i] = names[i].replace(cur_direc, "")
    faces_names.append(names[i])
end = time.time()
arrayForPickling=[]
arrayForPickling.append(faces_names)
arrayForPickling.append(faces_encodings)
with open('faces.pickle', 'wb') as f:
    pickle.dump(arrayForPickling, f)

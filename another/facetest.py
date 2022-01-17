import face_recognition
import cv2
import numpy as np
import os
import glob
from tqdm import tqdm
import time
import transliterate
import pickle
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing.image import img_to_array
import json
import tensorflow as tf
start = time.time()
faces_encodings = []
faces_names = []
end = time.time()

with open('faces.pickle', 'rb') as f:
    newdata = pickle.load(f)
faces_names=newdata[0]
faces_encodings=newdata[1]
for tmp in faces_names:
    print(str(tmp))
f = open("results.txt", "w")
f.write("Время для обработки лиц–"+str(end-start)+" секунд")
print("Время для обработки лиц–",end-start," секунд")
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
metadata=[]
fileloc="files/videos.mp4"
video_capture = cv2.VideoCapture(fileloc)
frame_width = int(video_capture.get(3))
metadata.append({"width":frame_width})
frame_height = int(video_capture.get(4))
metadata.append({"height":frame_height})
start = time.time()
fps = video_capture.get(cv2.CAP_PROP_FPS)
metadata.append({"fps":fps})
totalframes=video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
metadata.append({"total_frames":totalframes})
fileloc="processed"+fileloc
out = cv2.VideoWriter(fileloc,cv2.VideoWriter_fourcc(*'MP4V'), fps, (frame_width,frame_height))
pbar = tqdm(total=totalframes)
count=0
frames=[]
print(video_capture.isOpened())
while video_capture.isOpened():
    count=count+1
    ret, frame = video_capture.read()
    if not ret:
        break
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]
    if process_this_frame:
        face_locations = face_recognition.face_locations(small_frame)
        print(face_locations)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []


    pbar.update(1)

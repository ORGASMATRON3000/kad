#Определять все лица на изображении-DONE
#Для каждого лица считать им занимаемую площадь в процентах от общего изображения-DONE
#Распознавать людей на фотографиях-DONE
#Если человека нет в базе-записывать его и узнавать в будущем
#Распознавать эмоции
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
emotion_model_path = 'models/_mini_XCEPTION.91-0.64.hdf5'
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised","neutral"]
start = time.time()
faces_encodings = []
faces_names = []
"""
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

"""
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

video_capture = cv2.VideoCapture('testvid.mp4')
frame_width = int(video_capture.get(3))
frame_height = int(video_capture.get(4))
start = time.time()
# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
fps = video_capture.get(cv2.CAP_PROP_FPS)
totalframes=video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
print(fps)
print(totalframes)
print(totalframes/fps)
out = cv2.VideoWriter('output1.mp4',cv2.VideoWriter_fourcc(*'MP4V'), fps, (frame_width,frame_height))
pbar = tqdm(total=totalframes)
count=0
while video_capture.isOpened():
    count=count+1
    ret, frame = video_capture.read()
    if not ret:
        break
    small_frame = cv2.resize(frame, (0, 0), fx=1, fy=1)
    rgb_small_frame = small_frame[:, :, ::-1]
    if process_this_frame:
        face_locations = face_recognition.face_locations( rgb_small_frame)
        face_encodings = face_recognition.face_encodings( rgb_small_frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces (faces_encodings, face_encoding)
            name = "Неизвестно"
            face_distances = face_recognition.face_distance( faces_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = faces_names[best_match_index]
            else:
                tmpname=name+" "+str(len(faces_names))
                faces_names.append(tmpname)
                faces_encodings.append(face_encoding)
            face_names.append(name)
            f.write(str(name.replace('.jpg', '').replace('/face/', ''))+" появляется на "+ time.strftime('%H:%M:%S', time.gmtime(count/fps))+"\n")

    process_this_frame = not process_this_frame
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 1
        right *= 1
        bottom *= 1
        left *= 1
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        facesquare=abs(left-right)*abs(top-bottom)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi = gray[top:bottom,left:right]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        preds = emotion_classifier.predict(roi)[0]
        emotion_probability = np.max(preds)
        label = EMOTIONS[preds.argmax()]
        cv2.rectangle(frame, (left, bottom+35), (right, bottom+70), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, label, (left + 6, bottom+70 - 6), font, 1.0, (255, 255, 255), 1)

        height, width, channels = frame.shape
        pictsquare=height*width
        percent=100*facesquare/pictsquare
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, transliterate.translit(name.replace('.jpg', '').replace('/face/', ''), reversed=True), (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        cv2.rectangle(frame, (left, bottom), (right, bottom+35), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, "{:.2f}".format(percent)+"%", (left + 6, bottom+35 - 6), font, 1.0, (255, 255, 255), 1)

    pbar.update(1)
    """
    cv2.imshow("testframe",frame)
    if cv2.waitKey(1) == 27:
        break
    """
    out.write(frame)

end = time.time()
print("Время для обработки видео–",end-start," секунд")
#f.write("Время для обработки видео–"+str(end-start)+" секунд")
f.close()

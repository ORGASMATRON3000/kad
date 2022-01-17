import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from flask import Flask, request, abort, jsonify, send_from_directory


UPLOAD_DIRECTORY = "files"

if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)


api = Flask(__name__)

def ProcessVideo(fileloc,faceID,em):
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
    import os
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
    metadata=[]
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
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
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
                #f.write(str(name.replace('.jpg', '').replace('/face/', ''))+" появляется на "+ time.strftime('%H:%M:%S', time.gmtime(count/fps))+"\n")

        framemeta=[]
        process_this_frame = not process_this_frame
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
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
            #cv2.rectangle(frame, (left, bottom+35), (right, bottom+70), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            #cv2.putText(frame, label, (left + 6, bottom+70 - 6), font, 1.0, (255, 255, 255), 1)

            height, width, channels = frame.shape
            pictsquare=height*width
            percent=100*facesquare/pictsquare
            #cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            #cv2.putText(frame, transliterate.translit(name.replace('.jpg', '').replace('/face/', ''), reversed=True), (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            #cv2.rectangle(frame, (left, bottom), (right, bottom+35), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            #cv2.putText(frame, "{:.2f}".format(percent)+"%", (left + 6, bottom+35 - 6), font, 1.0, (255, 255, 255), 1)

            frame_data={
            "frame":count,
            "left":left,
            "top":top,
            "right":right,
            "bottom":bottom,
            "name":transliterate.translit(name.replace('.jpg', '').replace('/face/', ''), reversed=True),
            "percentage": "{:.2f}".format(percent),
            "emotion":label
            }
            #print(frame_data)
            framemeta.append(frame_data)


        frames.append({str(count):framemeta})

        pbar.update(1)
        out.write(frame)
    returnarr=[]
    returnarr.append({"metadata":metadata})
    returnarr.append({"frames":frames})
    returnjson = json.dumps(returnarr)
    end = time.time()
    print("Время для обработки видео–",end-start," секунд")
    #f.write("Время для обработки видео–"+str(end-start)+" секунд")
    f.close()
    return returnjson

def ConditionalProcess(fileloc,faceID,em,scale):
    #импорты всего нужного
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
    import os
    #подготовка к распознованию эмоций
    print("-----------------------------")
    print(type(em))
    print(em,faceID)
    em=em=="1"
    faceID=faceID=="1"
    print(em,faceID)
    print("-----------------------------")
    if em==True:
        emotion_model_path = 'models/_mini_XCEPTION.91-0.64.hdf5'
        emotion_classifier = load_model(emotion_model_path, compile=False)
        EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised","neutral"]
    else:
        print("NO")

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
    if faceID==True:
        with open('faces.pickle', 'rb') as f:
            newdata = pickle.load(f)
        faces_names=newdata[0]
        faces_encodings=newdata[1]
        for tmp in faces_names:
            print(str(tmp))

    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    metadata=[]
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
    #print(video_capture.isOpened())
    while video_capture.isOpened():
        count=count+1
        ret, frame = video_capture.read()
        if not ret:
            break
        small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        rgb_small_frame = small_frame[:, :, ::-1]
        if process_this_frame:
            if faceID==True:
                face_locations = face_recognition.face_locations(small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
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
            else:
                face_locations = face_recognition.face_locations(small_frame)
                face_names = []
                for face_location in face_locations:
                    name = "Неизвестно"
                    tmpname=name+" "+str(len(faces_names))
                    faces_names.append(tmpname)
                    face_names.append(name)

                #f.write(str(name.replace('.jpg', '').replace('/face/', ''))+" появляется на "+ time.strftime('%H:%M:%S', time.gmtime(count/fps))+"\n")

        framemeta=[]
        process_this_frame = not process_this_frame
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            reverceScale=float(1/scale)
            top *= reverceScale
            right *= reverceScale
            bottom *= reverceScale
            left *= reverceScale
            top=int(top)
            right=int(right)
            bottom=int(bottom)
            left=int(left)
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            facesquare=abs(left-right)*abs(top-bottom)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            roi = gray[top:bottom,left:right]
            roi = cv2.resize(roi, (48, 48))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            if em==True:
                preds = emotion_classifier.predict(roi)[0]
                emotion_probability = np.max(preds)
                label = EMOTIONS[preds.argmax()]
            else:
                label="Unknown"
            #cv2.rectangle(frame, (left, bottom+35), (right, bottom+70), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            #cv2.putText(frame, label, (left + 6, bottom+70 - 6), font, 1.0, (255, 255, 255), 1)

            height, width, channels = frame.shape
            pictsquare=height*width
            percent=100*facesquare/pictsquare
            #cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            #cv2.putText(frame, transliterate.translit(name.replace('.jpg', '').replace('/face/', ''), reversed=True), (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            #cv2.rectangle(frame, (left, bottom), (right, bottom+35), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            #cv2.putText(frame, "{:.2f}".format(percent)+"%", (left + 6, bottom+35 - 6), font, 1.0, (255, 255, 255), 1)

            frame_data={
            "frame":count,
            "left":left,
            "top":top,
            "right":right,
            "bottom":bottom,
            "name":transliterate.translit(name.replace('.jpg', '').replace('/face/', ''), reversed=True),
            "percentage": "{:.2f}".format(percent),
            "emotion":label
            }
            #print(frame_data)
            framemeta.append(frame_data)


        frames.append({str(count):framemeta})

        pbar.update(1)
        out.write(frame)
    returnarr=[]
    returnarr.append({"metadata":metadata})
    returnarr.append({"frames":frames})
    returnjson = json.dumps(returnarr)
    end = time.time()
    print("Время для обработки видео–",end-start," секунд")
    #f.write("Время для обработки видео–"+str(end-start)+" секунд")
    return returnjson
@api.route("/files")
def list_files():
    """Endpoint to list files on the server."""
    files = []
    for filename in os.listdir(UPLOAD_DIRECTORY):
        path = os.path.join(UPLOAD_DIRECTORY, filename)
        if os.path.isfile(path):
            files.append(filename)
    return jsonify(files)

@api.route("/files/<path:path>")
def get_file(path):
    """Download a file."""
    return send_from_directory(UPLOAD_DIRECTORY, path, as_attachment=True)


@api.route("/files/<filename>", methods=["POST"])
def post_file(filename):
    """Upload a file."""
    if "/" in filename:
        # Return 400 BAD REQUEST
        abort(400, "no subdirectories allowed")
    #print(request.args['target_URL'])
    faceID=1
    Emotions=1
    scale=1
    try:
        faceID=request.args['faceID']
    except:
        print ("faceID option was not prvided, setting to True")
    try:
        Emotions=request.args['Emotions']
    except:
        print ("Emotions option was not provided, setting to True")

    try:
        scale=float(request.args['scale'])
    except:
        print ("scale option was not provided, setting to 1")
    print(faceID,Emotions)
    with open(os.path.join(UPLOAD_DIRECTORY, filename), "wb") as fp:
        fp.write(request.data)
    print("faceID")
    print(faceID)
    print("Emotions")
    print(Emotions)
    print("scales")
    print(scale)
    return ConditionalProcess(os.path.join(UPLOAD_DIRECTORY, filename),faceID,Emotions,scale)


    # Return 201 CREATED
    return "", 201


if __name__ == "__main__":
    api.run(debug=True, host='0.0.0.0')

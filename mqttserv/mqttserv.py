from flask import Flask, request, abort, jsonify, send_from_directory
import face_recognition
import cv2
import numpy as np
import os
import glob
from tqdm import tqdm
import transliterate
import pickle
import paho.mqtt.client as mqtt #import the client1
import time
import threading
import json
import datetime

#functions for MQTT
def on_message(client, userdata, message):
    print("message received " ,str(message.payload.decode("utf-8")))
    #print("message topic=",message.topic)
    #print("message qos=",message.qos)
    #print("message retain flag=",message.retain)
    #pass
def on_log(client, userdata, level, buf):
    #print("log: ",buf)
    pass
#Defining Flask app
api = Flask(__name__)

#Global variables for thread mamgement
global START
global threadsStates

def videoProcess(rtsp,brocker,id):
    #Declaring global variables for thread managment
    global START
    global threadsStates

    #connecting to mqtt client
    broker_address=brocker
    print("creating new instance")
    client = mqtt.Client("P1") #create new instance
    client.on_log=on_log
    client.on_message=on_message #attach function to callback
    print("connecting to broker")
    client.connect(broker_address) #connect to broker
    client.loop_start() #start the loop
    print("Subscribing to topic","faceserver/records/")
    client.subscribe("faceserver/records/")

    #declaring variables for facial recognition
    faces_encodings = []
    faces_names = []


    #reading faces from /face directory
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



    #declaring varibales for face processing
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    metadata=[]
    seconds_in_day = 24 * 60 * 60

    #opening RTSP stream
    video_capture = cv2.VideoCapture(rtsp)
    frame_width = int(video_capture.get(3))
    metadata.append({"width":frame_width})
    frame_height = int(video_capture.get(4))
    metadata.append({"height":frame_height})
    start = time.time()
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    metadata.append({"fps":fps})
    totalframes=video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
    metadata.append({"total_frames":totalframes})
    count=0
    frames=[]
    print(video_capture.isOpened())
    facescur={}
    while video_capture.isOpened() and threadsStates[id][3]:
        try:
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
            #print("face_name- ",faces_names)


            framemeta=["framemeta"]
            process_this_frame = not process_this_frame


            for k in list(facescur.keys()):

                first_time =facescur[k]
                later_time = datetime.datetime.now()
                difference = later_time - first_time
                #datetime.timedelta(0, 8, 562000)
                difarr=divmod(difference.days * seconds_in_day + difference.seconds, 60)
                if difarr[0]>=60:
                    del facescur[k]




            for (top, right, bottom, left), name in zip(face_locations, face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                facesquare=abs(left-right)*abs(top-bottom)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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
                if name not in facescur:
                    facescur[name]=datetime.datetime.now()
                    frame_data={
                    "frame":count,
                    "left":left,
                    "top":top,
                    "right":right,
                    "bottom":bottom,
                    "name":transliterate.translit(name.replace('.jpg', '').replace('/face/', ''), reversed=True),
                    "percentage": "{:.2f}".format(percent)
                    }
                    framemeta.append(frame_data)
                #print(len(frame_data))
                #print("frame data-",frame_data)
                #client.publish("faceserver/records/",str(frame_data))
                #print(frame_data)

            if len(framemeta)>1:
                print("frame meta-",len(framemeta))
                print(facescur)
                #print("facecur- ", facescur)
                client.publish("faceserver/records/",str(framemeta))

        except BaseException as error:
            print('An exception occurred: {}'.format(error))
            return ("Not enough data")




        #frames.append({str(count):framemeta})

    client.loop_stop()
    threadsStates[id][3]=False
    print("STOPPED")



UPLOAD_DIRECTORY = "face"

@api.route("/command", methods=["POST"])
def start_proc():
    try:
        time.sleep(3)
        global threadsStates
        command=request.args.get("COMMAND")

        if command=="START":
            try:

                id=len(threadsStates)
                rtsp=request.args.get("RTSP")
                brocker=request.args.get("BROCKER")

                dataarr=[]
                dataarr.append(rtsp)
                dataarr.append(brocker)
                dataarr.append(id)
                dataarr.append(True)

                threadsStates.append(dataarr)


                x = threading.Thread(target=videoProcess, args=(rtsp,brocker,id,))
                x.start()


                return(str(id)+"-this process is starting")
            except BaseException as error:
                print('An exception occurred: {}'.format(error))
                return ("Not enough data")

        elif command=="STOP":
            id=request.args.get("ID")
            threadsStates[int(id)][3]=False
            return ("Stopping process "+str(id))
        elif command=="STATUS":
            return json.dumps(threadsStates)
        elif command=="UPLOAD":
            name=request.args.get("NAME")
            name=name+".jpg"
            with open(os.path.join(UPLOAD_DIRECTORY, name), "wb") as fp:
                fp.write(request.data)
            return "face uploaded"
        else:
            print("UNKNOWN COMMAND")
    except BaseException as error:
        print('An exception occurred: {}'.format(error))
        return "Command is not specified"


    return "", 201

@api.route("/face/<filename>", methods=["POST"])
def post_file(filename):
    if "/" in filename:
        # Return 400 BAD REQUEST
        abort(400, "no subdirectories allowed")
    filename=filename+".jpg"
    with open(os.path.join(UPLOAD_DIRECTORY, filename), "wb") as fp:
        fp.write(request.data)
    return "face uploaded"
if __name__ == "__main__":
    global threadsStates
    threadsStates=[]
    api.run(debug=True, host='0.0.0.0')
    #api.run(debug=True)

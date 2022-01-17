from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget
import cv2
import sys
from PyQt5.QtWidgets import  QMainWindow, QApplication, QWidget, QPushButton, QAction, QLineEdit, QMessageBox, QLabel,QPlainTextEdit, QVBoxLayout,QHBoxLayout
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot,QSize
from PyQt5.QtGui import QImage, QPixmap,QPalette,QBrush,QTextCursor
import numpy as np
import face_recognition
import glob
import time
import os
import threading
import pickle
from datetime import datetime
import encdecclass
import sys
from random import randint
import zlib
from base64 import urlsafe_b64encode as b64e, urlsafe_b64decode as b64d

def obscure(data: bytes) -> bytes:
    return b64e(zlib.compress(data, 9))

def unobscure(obscured: bytes) -> bytes:
    return zlib.decompress(b64d(obscured))

#Glovbal varibales for thread communication
global triggerShow
global triggerFaces
global img1,img2
#function that verifies identity
def checkSignature():
    file = open("iden.bin","rb")
    iden=file.read()
    file.close()
    file = open("signature.bin","rb")
    newsign=file.read()
    file.close()
    iden=iden.decode("utf-8")
    print(encdecclass.verifyMessage("TestCompany",iden,newsign))
    return encdecclass.verifyMessage("TestCompany",iden,newsign)
#function that returns log as a siries of continous entries
def fixLog(listOfNames,th):
    try:
        curarr=[]
        retlist=[]
        temps=[]
        temp=0
        count=0
        for i in range(len(listOfNames)-1):
            if (listOfNames[i][3]==True):
                continue
            for j in range(i+1,len(listOfNames)):
                if (listOfNames[j][3]==False and listOfNames[i][0]==listOfNames[j][0]):
                    diff=None

                    if curarr:
                        diff=datetime.strptime(str(listOfNames[j][1]-curarr[1][1]), "%H:%M:%S")
                    else:
                        diff=datetime.strptime(str(listOfNames[j][1]-listOfNames[i][1]), "%H:%M:%S")
                    if (diff<=th):
                        listOfNames[j][3]=True
                        if curarr:
                            curarr[1]=listOfNames[j]
                            temp=temp+float(listOfNames[j][2])
                            count=count+1
                        else:
                            curarr.append(listOfNames[i])
                            curarr.append(listOfNames[j])
                            temp=temp+float(listOfNames[j][2])
                            temp=temp+float(listOfNames[i][2])
                            coutn=count+2
            if curarr:
                retlist.append(curarr)
                curarr=[]
                #temps.append(temp/count)
                temp=0

        return retlist
    except:
        return 0
#function that removes errors from log
def deleteErrs(listOfNames,th):
    try:
        tmparr=[]
        for item in listOfNames:
            if datetime.strptime(str(item[1][1]-item[0][1]), "%H:%M:%S")>=th:
                tmparr.append(item)
        return tmparr
    except:
        return 0

#function that generates names for logs based on current time and date
def csvNameGen(name):
    try:
        tmp=datetime.now()
        return(name+str(tmp).replace(":","").replace("-","").replace(":",".")+".csv")
    except:
        return 0

#Save short log
def saveShortLog(listOfNames):
    try:
        import csv
        header=['Начало','Конец','Температура','Имя']
        with open(csvNameGen('shortLog'), 'w', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for item in listOfNames:
                data=[item[0][1].time().strftime('%H:%M:%S'),item[1][1].time().strftime('%H:%M:%S'),(float(item[0][2])+float(item[0][2]))/2,item[0][0]]
                writer.writerow(data)
    except:
        return 0


def saveFullLog(listOfNames):
    try:
        import csv
        header=['Время','Температура','Имя']
        with open(csvNameGen('longLog'), 'w', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for item in listOfNames:
                data=[item[1].time().strftime('%H:%M:%S'),item[2],item[0]]
                writer.writerow(data)
    except:
        return 0
def getTemp(frame):
    try:
        crop_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rows,cols=crop_img.shape
        summ=0
        count=0
        hth=255
        lth=180
        for i in range(rows):
            for j in range(cols):
                if (crop_img[i,j]<=hth and crop_img[i,j]>=lth):
                    summ=summ+crop_img[i,j]
                    count=count+1
        if count==0:
            return 0
        return(summ/count/15.3+24)
    except Exception as e:
        print("Temperature")
        print(str(e))
class logThread(QThread):
    received = pyqtSignal(str)
    try:
        def run(self):
            while True:
                time.sleep(3)
                lineList=[]
                with open("names.log", "r") as a_file:
                    for line in a_file:
                        stripped_line = line.strip()
                        lineList.append(stripped_line)
                if (len(lineList)<=0):
                    continue
                text=""
                if len(lineList)<=40:
                    for i in range(len(lineList)):
                        tmparr=lineList[len(lineList)-1-i].split(" ")
                        tmptext=tmparr[1]+" "+tmparr[2]+" "+tmparr[0]
                        text=text+tmptext+"\n"
                else:
                    for i in range((len(lineList)-40-1),len(lineList)):
                        tmparr=lineList[len(lineList)-1-(i-(len(lineList)-40-1))].split(" ")
                        tmptext=tmparr[1]+" "+tmparr[2]+" "+tmparr[0]
                        text=text+tmptext+"\n"

                self.received.emit(text)
    except:
        self.received.emit("Undefined")
def nameGen(name):
    try:
        os.makedirs("data/"+name+"/")
    except FileExistsError:
        # directory already exists
        pass
    dateTimeObj = datetime.now()
    timestampStr = dateTimeObj.strftime("%d-%b-%Y (%H:%M:%S.%f)")
    return ("data/"+name+"/"+timestampStr+name+".jpg")
class Thread(QThread):
    global triggerShow
    triggerShow=False
    changePixmap = pyqtSignal(QImage)
    def run(self):
        global img1,img2
        global triggerShow
        with open ("rtsp.config", "r") as myfile:
            data=myfile.readlines()
        for i in range(len(data)):
            data[i]=data[i].replace('\n', '')
        tmpTH=float(data[2])
        while True:
            print(triggerShow)
            if(triggerShow==False):
                rgbImage = np.zeros((700,700,3), np.uint8)
                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w
                convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
                p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                self.changePixmap.emit(p)
                time.sleep(1)
            else:
                #try:
                    print("Yahoo")
                    with open('faces_encodings.pickle', 'rb') as f:
                        faces_encodings = pickle.load(f)
                    with open('faces_names.pickle', 'rb') as f:
                        faces_names = pickle.load(f)

                    with open ("rtsp.config", "r") as myfile:
                        data=myfile.readlines()
                    for i in range(len(data)):
                        data[i]=data[i].replace('\n', '')
                    face_locations = []
                    face_encodings = []
                    face_names = []
                    process_this_frame = True
                    rtsp=data[0]
                    rtsp1=data[1]
                    """
                    print(rtsp)
                    print(rtsp1)
                    t1 = threading.Thread(target=RTSPget, args=(rtsp,7,1,0))
                    t2 = threading.Thread(target=RTSPget, args=(rtsp1,3,1,1))
                    t1.start()
                    t2.start()
                    print(rtsp,"Started")
                    """
                    cap = cv2.VideoCapture(data[0])
                    img1=None
                    img2=None
                    f = open("names.log", "wb")
                    vcapReg = cv2.VideoCapture(rtsp)
                    vcapTemp=cv2.VideoCapture(rtsp1)
                    ctReg=0
                    ctTemp=0
                    framesToSkip=7
                    while True:
                        if (triggerShow==True):
                            if (triggerShow==True):
                                ctReg += 1
                                ret = vcapReg.grab()
                                if ctReg % framesToSkip == 0: # skip some frames
                                    ret, frame = vcapReg.retrieve()
                                    if not ret: continue
                                    #frame = cv2.resize(frame, (0,0), fx=scale, fy=scale)
                                    img1=frame.copy()
                                ctTemp += 1
                                ret = vcapTemp.grab()
                                if ctTemp % framesToSkip == 0: # skip some frames
                                    ret, frame = vcapTemp.retrieve()
                                    if not ret: continue
                                    #frame = cv2.resize(frame, (0,0), fx=scale, fy=scale)
                                    img2=frame.copy()



                            else:
                                continue
                            if (img1 is not None and img2 is not None):
                                frame1=img1
                                frame2=img2
                                frame1 = cv2.resize(frame1, (0,0), fx=0.535, fy=0.56)
                                height1, width1, channels1 = frame1.shape
                                height2, width2, channels2 = frame2.shape
                                Xdiff=int(abs(width1-width2)/2)
                                Ydiff=int(abs(height1-height2)/2)

                                frame1 = frame1[0+Ydiff+1:height1-1-Ydiff+1, 0+Xdiff-10:width1-1-Xdiff-10]

                                small_frame = cv2.resize(frame1, (0, 0), fx=0.5, fy=0.5)
                                rgb_small_frame = small_frame[:, :, ::-1]
                                namestemps=[]
                                if process_this_frame:
                                    face_locations = face_recognition.face_locations(rgb_small_frame)
                                    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                                    face_names = []
                                    for face_encoding in face_encodings:
                                        matches = face_recognition.compare_faces (faces_encodings, face_encoding)
                                        name = "Unknown"
                                        face_distances = face_recognition.face_distance( faces_encodings, face_encoding)
                                        best_match_index = np.argmin(face_distances)
                                        if matches[best_match_index]:
                                            name = faces_names[best_match_index]
                                        face_names.append(name)

                                        now = datetime.now()
                                        current_time = now.strftime("%H:%M:%S")
                                        namestemps.append([str(name.replace('.jpg', '').replace("\\face\\", '')),str(current_time),str(0)])

                                process_this_frame = not process_this_frame
                                count=0
                                for (top, right, bottom, left), name in zip(face_locations, face_names):
                                    top *= 2
                                    right *= 2
                                    bottom *= 2
                                    left *= 2
                                    temperature=getTemp(frame1[top:bottom, left:right])
                                    if float(temperature)>tmpTH:
                                        pictname=nameGen(name.replace('\\face\\', ''))
                                        print(pictname)
                                        cv2.imwrite(pictname,frame1[top:bottom,left:right])

                                    cv2.rectangle(frame1, (left, top), (right, bottom), (0, 0, 255), 2)
                                    cv2.rectangle(frame1, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                                    cv2.rectangle(frame1, (left, bottom), (right, bottom+35), (0, 0, 255), cv2.FILLED)
                                    cv2.rectangle(frame2, (left, top), (right, bottom), (0, 0, 255), 2)
                                    cv2.rectangle(frame2, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                                    font = cv2.FONT_HERSHEY_DUPLEX

                                    if(len(namestemps)>0):
                                        namestemps[count][2]=str(temperature)

                                    cv2.putText(frame1, name.replace('.jpg', '').replace('\\face\\', ''), (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
                                    cv2.putText(frame1, str("%.2f" %temperature), (left + 6, bottom - 6+35), font, 0.5, (255, 255, 255), 1)
                                    count=count+1

                                for element in namestemps:
                                    f = open("names.log", "a")
                                    f.write(str(element[0])+" "+str(element[1])+" "+str(element[2])[0:4]+"\n")
                                x,y,z=frame1.shape
                                frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
                                h, w, ch = frame1.shape
                                bytesPerLine = ch * w
                                convertToQtFormat = QImage(frame1.data, w, h, bytesPerLine, QImage.Format_RGB888)
                                p = convertToQtFormat.scaled(700, 700, Qt.KeepAspectRatio)
                                self.changePixmap.emit(p)
                        else:
                            break
                            """
                except Exception as e:
                    print("Video Getting")
                    print(str(e))
                    triggerShow=False
"""



class AnotherWindow(QWidget):
    """
    This "window" is a QWidget. If it has no parent, it
    will appear as a free-floating window as we want.
    """
    @pyqtSlot(QImage)
    def setImage(self, image):
        self.label.setPixmap(QPixmap.fromImage(image))

    def __init__(self):
        super().__init__()

        layout = QVBoxLayout()
        layoutStartStop=QHBoxLayout()
        layoutLOG=QVBoxLayout()
        # Create textbox
        self.textedit = QPlainTextEdit(self)
        self.textedit.resize(300,500)
        self.textedit.setObjectName("log")
        self.textedit.setReadOnly(True)
        layoutLOG.addWidget(self.textedit)
        t = logThread(self)
        t.received.connect(self.textedit.setPlainText)
        t.start()

        #create button
        self.button = QPushButton('Сохранить лог', self)
        self.button.clicked.connect(self.saveLog)
        layoutLOG.addWidget(self.button)
        th = Thread(self)
        th.changePixmap.connect(self.setImage)
        th.start()
        self.label = QLabel("Another Window % d" % randint(0,100))
        layout.addWidget(self.label)
        # Create a button in the window


        self.button = QPushButton('Запуск стрима', self)
        self.button.clicked.connect(self.on_click)
        layoutStartStop.addWidget(self.button)
        self.button = QPushButton('Остановка стрима', self)
        self.button.clicked.connect(self.stopStream)
        layoutStartStop.addWidget(self.button)


        layout.addLayout(layoutStartStop)
        layout.addLayout(layoutLOG)
        self.setLayout(layout)
    @pyqtSlot()
    def saveLog(self):
        try:
            from datetime import datetime
            lineList=[]
            with open("names.log", "r") as a_file:
              for line in a_file:
                stripped_line = line.strip()
                tmp=stripped_line.split(" ")
                tmp.append(False)
                tmp[1]=datetime.strptime(tmp[1],'%H:%M:%S')
                lineList.append(tmp)
            saveFullLog(lineList)

            trashhold=datetime.strptime("00:00:05",'%H:%M:%S')
            xex=fixLog(lineList,trashhold)


            length=datetime.strptime("00:00:03",'%H:%M:%S')
            mem=deleteErrs(xex, length)
            saveShortLog(mem)
        except Exception as e:
            print("saving log")
            print(str(e))
    @pyqtSlot()
    def on_click(self):
        global triggerShow
        try:
            if (triggerShow!=True):

                triggerShow=True
                #print(triggerShow)
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Information)
                msg.setText("Запускакем стрим,пожалйста подождите...")
                msg.setWindowTitle("Запуск стрима")
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()
                triggerShow=True

        except Exception as e:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText(str(e))
            msg.setWindowTitle("Ошибка лицензии")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            """
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText("Не удалось запустить стрим")
            msg.setWindowTitle("Ошибка")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            """
            triggerShow=False


#Stop stream
    @pyqtSlot()
    def stopStream(self):
        global triggerShow
        if (triggerShow!=False):
            triggerShow=False


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        with open ("rtsp.config", "r") as myfile:
            data=myfile.readlines()
        for i in range(len(data)):
            data[i]=data[i].replace('\n', '')
        #create button
        layout = QVBoxLayout()
        #create textbox
        self.textbox = QLineEdit(self)
        self.textbox.resize(400,20)
        self.textbox.setText(data[0])
        self.textbox.setObjectName("rtsp1")
        layout.addWidget(self.textbox)

        #create QLineEdit
        self.textbox = QLineEdit(self)
        self.textbox.resize(400,20)
        self.textbox.setText(data[1])
        self.textbox.setObjectName("rtsp2")
        layout.addWidget(self.textbox)

        self.button = QPushButton('Обработать лица', self)
        self.button.move(20,550)
        self.button.clicked.connect(self.process_Faces)
        layout.addWidget(self.button)

        self.w = None  # No external window yet.
        self.button = QPushButton("Запуск тепловизра")
        self.button.clicked.connect(self.show_new_window)
        layout.addWidget(self.button)

        self.button = QPushButton('Cгенерировать идентификатор', self)
        self.button.clicked.connect(self.generate_iden)
        layout.addWidget(self.button)

        wid = QWidget(self)
        self.setCentralWidget(wid)
        wid.setLayout(layout)
    @pyqtSlot()
    def generate_iden(self):
        file = open("iden.bin","wb")
        rtsp1=self.findChild(QLineEdit, "rtsp1")
        rtsp2=self.findChild(QLineEdit, "rtsp2")
        rtspstring=rtsp1.text()+rtsp2.text()
        file.write(obscure(rtspstring.encode('utf-8')))
        file.close()

    def show_new_window(self, checked):
        try:
            self.generate_iden()
            print("License ",checkSignature())
            if checkSignature():
                with open ("rtsp.config", "r") as myfile:
                    data=myfile.readlines()
                for i in range(len(data)):
                    data[i]=data[i].replace('\n', '')

                rtsp1=self.findChild(QLineEdit, "rtsp1")
                rtsp2=self.findChild(QLineEdit, "rtsp2")
                with open('rtsp.config', 'w') as the_file:
                    the_file.write(rtsp1.text()+'\n')
                    the_file.write(rtsp2.text()+'\n')
                    the_file.write(data[2])

                if self.w is None:
                    self.w = AnotherWindow()
                self.w.show()
            else:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Information)
                s="Ошибка лицензии"
                msg.setText(s)
                msg.setWindowTitle("Ошибка лицензии")
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()
        except Exception as e:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText(str(e))
            msg.setWindowTitle("Ошибка при проверке")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()


    def faceThread(self):
        global triggerShow
        if (triggerShow==False):
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
            with open('faces_encodings.pickle', 'wb') as f:
                pickle.dump(faces_encodings, f)

            with open('faces_names.pickle', 'wb') as f:
                pickle.dump(faces_names, f)
            print("Done")



    @pyqtSlot()
    def process_Faces(self):
        global triggerShow
        if triggerShow==False:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            s="Начинаем обработку лиц, подождите..."
            msg.setText(s)
            msg.setWindowTitle("Обработка лиц")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            t = threading.Thread(target = self.faceThread, args = ())
            t.start()
        else:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText("Невозможно обработать лица пока идет стрим")
            msg.setWindowTitle("Ошибка")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()


app = QApplication(sys.argv)
w = MainWindow()
w.show()
app.exec()

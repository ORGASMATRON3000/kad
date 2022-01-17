FROM server-template
ADD . / network/
WORKDIR /network
#RUN apt-get update && apt-get -y install cmake protobuf-compiler vim
#RUN apt-get install ffmpeg libsm6 libxext6  -y
#RUN pip install opencv-contrib-python==4.5.3.56 opencv-python==4.5.1.48 tqdm transliterate tensorflow==2.5.0 Flask face-recognition

CMD ["python", "main.py"]

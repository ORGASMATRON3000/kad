import cv2

video_capture = cv2.VideoCapture("rtsp://192.168.0.107/emotions.mkv")
print(video_capture.isOpened())
# Read until video is completed
while(video_capture.isOpened()):
  # Capture frame-by-frame
  ret, frame = video_capture.read()
  if ret == True:

    # Display the resulting frame
    cv2.imshow('Frame',frame)

    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break

  # Break the loop
  else:
    break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()

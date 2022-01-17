import vlc
import time
player=vlc.MediaPlayer('rtsp://192.168.102.6:554/0')
player.play()
while 1:
    time.sleep(10000)
    player.video_take_snapshot(0, '.snapshot.tmp.png', 0, 0)

import paho.mqtt.client as mqtt #import the client1
import time
############
def on_message(client, userdata, message):
    print("message received " ,str(message.payload.decode("utf-8")))
    print("message topic=",message.topic)
    print("message qos=",message.qos)
    print("message retain flag=",message.retain)
def on_log(client, userdata, level, buf):
    print("log: ",buf)
########################################
broker_address="doccloud.ru"
#broker_address="iot.eclipse.org"
print("creating new instance")
client = mqtt.Client("P1") #create new instance
client.on_log=on_log
client.on_message=on_message #attach function to callback
print("connecting to broker")
client.connect(broker_address) #connect to broker
client.loop_start() #start the loop
print("Subscribing to topic","faceserver/records/")
client.subscribe("faceserver/records/")
print("Publishing message to topic","faceserver/records/")
while True:
    client.publish("faceserver/records/","TEST")
    time.sleep(4)
time.sleep(4) # wait
client.loop_stop() #stop the loop

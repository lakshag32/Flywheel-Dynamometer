#https://stackoverflow.com/questions/42339876/error-unicodedecodeerror-utf-8-codec-cant-decode-byte-0xff-in-position-0-in
#https://www.youtube.com/watch?v=VN3HJm3spRE

import serial
import time
import csv 

arduino = serial.Serial(port='COM4',  baudrate=9600, timeout=.1)
time.sleep(1)

while True: 
    #while there no data at the serial port
    while(arduino.in_waiting == 0):
        pass

    data = str(arduino.readline()," ISO-8859-1").strip("\r\n")
   
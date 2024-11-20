import serial

ser = serial.Serial('COM4', 9600)  # Replace 'COM1' with your port and 9600 with your baud rate

while True:
    if ser.in_waiting > 0:
        line = ser.readline().decode('utf-8').rstrip()
        try:
            value = int(line)
            print(value)
        except:
            print("Invalid data received:", line)
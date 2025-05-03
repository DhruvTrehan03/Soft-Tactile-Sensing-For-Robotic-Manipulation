"""
Information:
Serial Number for Right fingertip: 14664510
Serial Number for Left fingertip: 14664070
Serial Number for Arduino Uno: 85036313530351C0A160

"""


import serial
import serial.tools.list_ports
import time

ports = list(serial.tools.list_ports.comports())
for p in ports:
    print(p)

def serial_connect(serial_number):
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        if p.serial_number == serial_number:
            return serial.Serial('COM12', baudrate=115200, timeout=.1)

arduino = serial_connect('85036313530351C0A160')
for i in range(100):
    data = arduino.readline().strip().decode('utf-8')
    if data.isnumeric():
        print(data)
        time.sleep(0.05)

ports = list(serial.tools.list_ports.comports())
for p in ports:
    print(p)

arduino.close()
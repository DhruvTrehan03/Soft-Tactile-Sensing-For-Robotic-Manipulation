"""
Information:
Serial Number for Right fingertip: 14664510
Serial Number for Left fingertip: 14664070
Serial Number for Arduino Uno: 85036313530351C0A160

"""




import time
import serial
import serial.tools.list_ports

import threading
# import numpy
# import socket
# from math import pi
from pyRobotiqGripper import RobotiqGripper
import time
from pyRobotiqGripper import RobotiqGripper
import time

import waypoints as wp
import kg_robot as kgr

def begin(burt,gripper):
    print(burt.getj()) #[0.139679, 0.607655, 0.397171, 3.07663, -0.177625, 0.0407474] directly above screwdriver
    burt.movel([0.139679, 0.607655, 0.397171, 3.07663, -0.177625, 0.0407474], acc=0.05, vel=0.05)
    gripper.goTomm(50, 50, 100) #open to 50mm
    position_in_mm = gripper.getPositionmm()
    print(position_in_mm)
    time.sleep(0.5)

    burt.translatel_rel([0, 0, -0.1, 0, 0, 0], acc=0.05, vel=0.05) #translate this much to bring around screwdrive
    time.sleep(0.5)

def rotate (burt,gripper, rotation):
    gripper.goTomm(30, 50, 100) #close to 30mm
    position_in_mm = gripper.getPositionmm()
    print(position_in_mm)
    time.sleep(0.5)

    pre_rot = burt.getj()

    print(pre_rot)
    burt.movej_rel([0, 0, 0, 0, 0, rotation], acc=0.1, vel=0.1) #rotate 90 degrees approx clockwise (this lightly touches/reaches load cell)
    time.sleep(0.5)

    burt.movej(pre_rot, acc=0.1, vel=0.1) #rotate 90 degrees approx counter-clockwise
    time.sleep(0.5)
    print(burt.getj())

def reset (burt,gripper):
    gripper.goTomm(50, 50, 100) #open to 50mm
    position_in_mm = gripper.getPositionmm()
    print(position_in_mm)
    time.sleep(0.5)

    burt.translatel_rel([0, 0, 0.1, 0, 0, 0], acc=0.05, vel=0.05) #translate this much to bring above screwdriver

    print(burt.getj())

def setup():
    print("------------Configuring Burt-------------\r\n")
    burt = kgr.kg_robot(port=30010,db_host="169.254.68.20") #Setup UR5
    
    gripper = RobotiqGripper() #Setup Gripper
    gripper.activate()
    time.sleep(0.5)

    gripper.calibrate(0, 120) #calibrate
    print("calibrated")
    time.sleep(0.5)

    load = serial_connect('85036313530351C0A160')
    left_finger = serial_connect('14664070')
    right_finger = serial_connect('14664510')

    return burt,gripper,load, left_finger,right_finger

def serial_connect(serial_number):
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        if p.serial_number == serial_number:
            return serial.Serial(p.device)

def read_data(device):
    data = device.readline().strip().decode('utf-8')
    time.sleep(0.05)
    if data.isnumeric():
        return data

def main():

    burt,gripper,load,left_finger,right_finger = setup() #Setup All Devices
    begin(burt,gripper)
    for i in range(1):
        print(f'-----{i}------')
        rotate(burt, gripper, 1.29)
    
    reset(burt,gripper)
    
    burt.close()
if __name__ == '__main__': main()


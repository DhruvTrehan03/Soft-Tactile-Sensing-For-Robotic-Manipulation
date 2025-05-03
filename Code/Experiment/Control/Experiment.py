"""
Information:
Serial Number for Right fingertip: 14664510
Serial Number for Left fingertip: 14664070
Serial Number for Arduino Uno: 85036313530351C0A160

"""

import time
from datetime import datetime




import numpy as np
# import socket
# from math import pi

from pyRobotiqGripper import RobotiqGripper

import waypoints as wp
import kg_robot as kgr

def setup():
    print("------------Configuring Burt-------------\r\n")
    burt = kgr.kg_robot(port=30010,db_host="169.254.68.20") #Setup UR5
    print(burt.getl())


    gripper = RobotiqGripper() #Setup Gripper
    gripper.activate()
    time.sleep(0.5)

    gripper.calibrate(0, 120) #calibrate
    print("calibrated")
    time.sleep(0.5)

    return burt,gripper

def begin(burt,gripper):
    print(burt.getj()) #[0.139679, 0.607655, 0.397171, 3.07663, -0.177625, 0.0407474] directly above screwdriver
    burt.movel([0.129806, 0.608642, 0.397123, 3.07652, -0.177733, 0.0408581], acc=0.05, vel=0.05)
    gripper.goTomm(50, 50, 100) #open to 50mm
    position_in_mm = gripper.getPositionmm()
    time.sleep(0.5)

    burt.translatel_rel([0, 0, -0.1, 0, 0, 0], acc=0.05, vel=0.05) #translate this much to bring around screwdrive
    time.sleep(0.5)


def reset (burt,gripper):
    gripper.goTomm(50, 50, 100) #open to 50mm
    position_in_mm = gripper.getPositionmm()
    time.sleep(0.5)

    burt.translatel_rel([0, 0, 0.1, 0, 0, 0], acc=0.05, vel=0.05) #translate this much to bring above screwdriver

    print(burt.getj())

def rotate (burt,gripper, rotation):
    gripper.goTomm(25, 50, 100) #close to 30mm
    position_in_mm = gripper.getPositionmm()
    time.sleep(0.5)

    pre_rot = burt.getj()
    burt.movej_rel([0, 0, 0, 0, 0, rotation], acc=0.1, vel=0.1) #rotate 90 degrees approx clockwise (this lightly touches/reaches load cell)
    time.sleep(0.5)

    burt.movej(pre_rot, acc=0.1, vel=0.1) #rotate 90 degrees approx counter-clockwise
    time.sleep(0.5)

def rotate_varying(burt, gripper,start,end,number):

    gripper.goTomm(25, 50, 100) #close to 30mm
    position_in_mm = gripper.getPositionmm()
    time.sleep(0.5)

    pre_rot = burt.getj()
    rotations = np.linspace(float(start), float(end), int(number))
    print(rotations)
    for rotation in rotations:
        burt.movej_rel([0, 0, 0, 0, 0, rotation], acc=0.1, vel=0.1) #rotate 90 degrees approx clockwise (this lightly touches/reaches load cell)
        time.sleep(0.5)

        burt.movej(pre_rot, acc=0.1, vel=0.1) #rotate 90 degrees approx counter-clockwise
        time.sleep(0.5)

def main():
    print(datetime.now().strftime("%d-%m-%Y_%H-%M")) 
    # f = open(f'C:\\Users\\dhruv\\4th Year Project\\MATLAB\\Large_Data\\{datetime.now().strftime("%Y-%m-%d_%H-%M")}\\Motion.txt', "w")
    burt,gripper = setup() #Setup All Devices

    begin(burt,gripper)
    for i in range(10):
        print(i)
        # f.write(f'{datetime.now().time()}, {i}\n')
        rotate_varying(burt, gripper, 1.45,1.65,10)
    
    reset(burt,gripper)
    # f.close()
    burt.close()

if __name__ == '__main__':
    main()
import time
from datetime import datetime
import numpy as np
from pyRobotiqGripper import RobotiqGripper
import kg_robot as kgr

pos40 = [0.491371, 0.467741, 0.33044, 2.23535, -2.16975, 0.0159733]  #0.1 above [0.491371, 0.467741,  0.23044, 2.23535, -2.16975, 0.0159733]
pos30 =[0.34721, 0.395927, 0.33044, 2.23527, -2.16976, 0.0158725] #0.1 above [0.34721, 0.395927,  0.23044, 2.23527, -2.16976, 0.0158725]
pos20 = [0.24164, 0.482094, 0.318499, 2.23526, -2.1698, 0.0160571] #0.1 above [0.24164, 0.482094, 0.218499, 2.23526, -2.1698, 0.0160571]
pos10 = [0.118385, 0.386157, 0.331611, 2.2354, -2.16973, 0.015951] #0.1 above [0.118385, 0.386157, 0.231611, 2.2354, -2.16973, 0.015951]

def setup_burt(port=30010, db_host="169.254.68.20"):
    print("------------Configuring Burt-------------")
    burt = kgr.kg_robot(port=port, db_host=db_host)
    print(f"Initial position: {burt.getl()}")

    # burt.movel((pos10), acc=0.05, vel=0.05) #Move to 10mm position
    # time.sleep(0.5)
    # burt.movel((pos20), acc=0.05, vel=0.05) #Move to 20mm position
    # time.sleep(0.5)
    # burt.movel((pos30), acc=0.05, vel=0.05) #Move to 30mm position
    # time.sleep(0.5)
    # burt.movel((pos40), acc=0.05, vel=0.05) #Move to 40mm position
    # time.sleep(0.5)
    # burt.movel((pos30), acc=0.05, vel=0.05) #Move to 30mm position
    # time.sleep(0.5)
    # burt.movel((pos20), acc=0.05, vel=0.05) #Move to 20mm position
    # time.sleep(0.5)
    # burt.movel((pos10), acc=0.05, vel=0.05) #Move to 10mm position
    # time.sleep(0.5)
    
    #burt.movel([0.278787, 0.293363, 0.432748, 2.161, -2.16355, 0.0448998], acc=0.05, vel=0.05) #Grab Screwdriver final position is [0.378787, 0.293363, 0.332748, 2.161, -2.16355, 0.0448998] difference is [0.2, 0, -0.2, 0, 0, 0]
    return burt


def setup_gripper():
    gripper = RobotiqGripper()
    gripper.activate()
    time.sleep(0.5)
    gripper.calibrate(0, 120)
    print("Gripper calibrated")
    return gripper

def initialise(burt, gripper):
    print("Starting movement sequence...")
    gripper.goTomm(50, 50, 100)
    time.sleep(0.5)
    #burt.translatel_rel([0, 0, -0.1, 0, 0, 0], acc=0.05, vel=0.05)
    time.sleep(0.5)

def initialise_demo(burt):
    print("Starting movement sequence...")
    # gripper.goTomm(50, 50, 100)
    time.sleep(0.5)
    burt.movel([0.378787, 0.293363, 0.432748, 2.161, -2.16355, 0.0448998], acc=0.05, vel=0.05) #Grab Screwdriver
    time.sleep(1)
    # gripper.goTomm(25, 50, 100)
    time.sleep(0.5)

def rotate_varying(burt, gripper, start, end, steps):
    rotations = np.linspace(float(start), float(end), int(steps))
    pre_rot = burt.getj()
    gripper.goTomm(22, 50, 100) #diamerter, force, speed
    for rotation in rotations:
        burt.movej_rel([0, 0, 0, 0, 0, rotation], acc=0.1, vel=0.1)
        time.sleep(5)
        burt.movej(pre_rot, acc=0.1, vel=0.1)
        time.sleep(0.5)

def rotate_demo(burt):
    burt.movel([0.378787, 0.293363, 0.332748, 2.161, -2.16355, 0.0448998], acc=0.05, vel=0.05) 
    time.sleep(0.5)
    pre_rot = burt.getj()
    burt.movej_rel([0, 0, 0, 0, 0, 1], acc=0.1, vel=0.2)
    time.sleep(0.5)
    burt.movej(pre_rot, acc=0.1, vel=0.1)
    time.sleep(0.5)
    #invert translation
    burt.translatel_rel([0, 0, 0.05, 0, 0, 0], acc=0.05, vel=0.05)
    time.sleep(0.5)
    burt.movel([0.278787, 0.293363, 0.432748, 2.161, -2.16355, 0.0448998], acc=0.05, vel=0.05) #Grab Screwdriver
    time.sleep(0.5)



def main_experiment(save_file=False, use_burt=True, use_gripper=True, name="Motion"):
    # Initialize optional file saving
    log_file = None
    if save_file:
        filename = f"{name}_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.txt"
        log_file = open(filename, "w")
        print(f"Logging to file: {filename}")

    burt = setup_burt() if use_burt else None
    gripper = setup_gripper() if use_gripper else None


    if burt and gripper:
        initialise(burt, gripper)
        time.sleep(1)
        for i in range(5): #Number of loops
            #Log when each action occurs
            print(f"Iteration {i}")
            if log_file:
                log_file.write(f"{datetime.now().time()}, Iteration {i}\n")
            #Action
            rotate_varying(burt,gripper, 0.08, 0.08, 1)

        # Reset burt and gripper
        gripper.goTomm(50, 50, 100)
        #burt.translatel_rel([0, 0, 0.1, 0, 0, 0], acc=0.05, vel=0.05)
        print(f"Reset position: {burt.getj()}")

    # Cleanup
    if burt:
        burt.close()
    if gripper:
        gripper = gripper.reset()  # Assume gripper has no explicit close method
    if log_file:
        log_file.close()

def main_demo(use_burt=True, use_gripper=True):
    burt = setup_burt() if use_burt else None
    time.sleep(2)
    gripper = setup_gripper() if use_gripper else None

    initialise_demo(burt)
    rotate_demo(burt)

    # Reset burt and gripper
    # gripper.goTomm(50, 50, 100)
    # burt.translatel_rel([0, 0, 0.1, 0, 0, 0], acc=0.05, vel=0.05)
    print(f"Reset position: {burt.getj()}")

    # Cleanup
    if burt:
        burt.close()
    if gripper:
        gripper = gripper.reset()

if __name__ == "__main__":
    main_experiment(save_file=False, use_burt=True, use_gripper=True, name="20mm")
    #main_demo(use_burt=True, use_gripper=False)

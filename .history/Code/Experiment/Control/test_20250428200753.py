from pyRobotiqGripper import RobotiqGripper
import time
gripper = RobotiqGripper()  # Replace 'COM3' with the correct port
gripper.reset()  # Reset the gripper to its initial state
time.sleep(1)  # Wait for the gripper to reset
gripper.activate()
time.sleep(1)  # Wait for the gripper to activate
gripper.calibrate(0,50)  # Calibrate the gripper
print("Gripper activated successfully")

from pyRobotiqGripper import RobotiqGripper
import time
gripper = RobotiqGripper()  # Replace 'COM3' with the correct port
try:
    gripper.activate()
    time.sleep(2)  # Wait for the gripper to activate
    print("Gripper activated successfully")
except Exception as e:
    print(f"Error: {e}")
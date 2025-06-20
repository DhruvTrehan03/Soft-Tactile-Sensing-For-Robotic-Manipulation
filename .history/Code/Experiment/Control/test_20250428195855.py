from pyRobotiqGripper import RobotiqGripper
import time
gripper = RobotiqGripper()  # Replace 'COM3' with the correct port
try:
    gripper.reset()  # Reset the gripper to its initial state
    time.sleep(1)  # Wait for the gripper to reset
    gripper.activate()
    time.sleep(1)  # Wait for the gripper to activate
    print("Gripper activated successfully")
except Exception as e:
    print(f"Error: {e}")
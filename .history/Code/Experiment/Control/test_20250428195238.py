from pyRobotiqGripper import RobotiqGripper

gripper = RobotiqGripper()  # Replace 'COM3' with the correct port
try:
    gripper.activate()
    print("Gripper activated successfully")
except Exception as e:
    print(f"Error: {e}")
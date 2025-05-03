from pyRobotiqGripper import RobotiqGripper
import time

gripper = RobotiqGripper()
gripper.activate()

print('Gripper Found')
print('Gripper Activated')
time.sleep(0.5)
gripper.calibrate(0, 120)
print("calibrated")
time.sleep(0.5)

gripper.goTomm(50, 50, 100)
position_in_mm = gripper.getPositionmm()
print(position_in_mm)
time.sleep(5)

gripper.goTomm(25, 50, 100)
position_in_mm = gripper.getPositionmm()
print(position_in_mm)
time.sleep(50)

gripper.reset()
# gripper.goTomm(30, 50, 100)
# position_in_mm = gripper.getPositionmm()
# print(position_in_mm)

# for i in range(2):

#     gripper.goTo(120,255,10)
#     position_in_bit = gripper.getPosition()
#     print(position_in_bit)
#     time.sleep(0.5)

#     gripper.goTomm(20, 50, 100)
#     position_in_mm = gripper.getPositionmm()
#     print(position_in_mm)
#     time.sleep(2)

# gripper.goTomm(120,50,255)
# position_in_mm = gripper.getPositionmm()
# print(position_in_mm)
"""
Information:
Serial Number for Right fingertip: 14664510
Serial Number for Left fingertip: 14664070
Serial Number for Arduino Uno: 85036313530351C0A160
"""

import serial
import serial.tools.list_ports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.preprocessing import MinMaxScaler

MAX_FRAMES = 100  # Number of time frames to maintain

def serial_connect(serial_number):
    """
    Establish a serial connection to the device with the given serial number.
    """
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        if p.serial_number == serial_number:
            return serial.Serial(p.device, 115200, timeout=25)
    print(f"Device with serial number {serial_number} not found.")  # Error handling
    return None  # Return None if device is not found

# Setup serial connections for the devices
left = serial_connect('14664070')
right = serial_connect('14664510')

if left is None or right is None:
    raise RuntimeError("Could not connect to one or more devices.")

# Send a "y" to the devices to start communication
left.write(b'y')
right.write(b'y')

# Determine the number of electrode pairings dynamically
print("Reading initial data to determine the number of electrode pairings...")
initial_data1 = left.readline().decode('utf-8').strip().split(',')
initial_data2 = right.readline().decode('utf-8').strip().split(',')

if not initial_data1 or not initial_data2:
    raise RuntimeError("No data received from devices. Check the connections.")

NUM_PAIRINGS1 = len(initial_data1)
NUM_PAIRINGS2 = len(initial_data2)

print(f"Device 1 has {NUM_PAIRINGS1} electrode pairings.")
print(f"Device 2 has {NUM_PAIRINGS2} electrode pairings.")

# Initialize data buffers
plotthis1 = np.zeros((MAX_FRAMES, NUM_PAIRINGS1))  # Buffer for Device 1
plotthis2 = np.zeros((MAX_FRAMES, NUM_PAIRINGS2))  # Buffer for Device 2

# Initialize the plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
lines1 = [ax1.plot([], [], label=f"Pairing {i+1}")[0] for i in range(NUM_PAIRINGS1)]  # Lines for Device 1
lines2 = [ax2.plot([], [], label=f"Pairing {i+1}")[0] for i in range(NUM_PAIRINGS2)]  # Lines for Device 2

# Configure subplots
ax1.set_title("Real-Time Data from Device 1")
ax1.set_xlabel("Time Frames")
ax1.set_ylabel("Magnitude")


ax2.set_title("Real-Time Data from Device 2")
ax2.set_xlabel("Time Frames")
ax2.set_ylabel("Magnitude")


scaler1 = MinMaxScaler(feature_range=(0, 1))  # Scaler for Device 1
scaler2 = MinMaxScaler(feature_range=(0, 1))  # Scaler for Device 2

# Function to update the plot
def update_plot(frame):
    global plotthis1, plotthis2
    
    # Read data from the devices
    try:
        data1 = left.readline().decode('utf-8').strip().split(',')
        data2 = right.readline().decode('utf-8').strip().split(',')
        
        # Ensure data is in the correct format
        if len(data1) == NUM_PAIRINGS1 and len(data2) == NUM_PAIRINGS2:
            data1 = np.array(data1, dtype=float)  # Convert to NumPy array
            data2 = np.array(data2, dtype=float)
            
            # Normalize the data
            data1 = scaler1.fit_transform(data1.reshape(-1, 1)).flatten()
            data2 = scaler2.fit_transform(data2.reshape(-1, 1)).flatten()
            
            # Update the buffers
            plotthis1 = np.roll(plotthis1, -1, axis=0)  # Shift rows up
            plotthis2 = np.roll(plotthis2, -1, axis=0)
            plotthis1[-1, :] = data1  # Add new data to the last row
            plotthis2[-1, :] = data2
            
            # Update the lines for Device 1
            for i, line in enumerate(lines1):
                line.set_data(range(MAX_FRAMES), plotthis1[:, i])
            
            # Update the lines for Device 2
            for i, line in enumerate(lines2):
                line.set_data(range(MAX_FRAMES), plotthis2[:, i])
            
            # Set axis limits dynamically
            ax1.set_xlim(0, MAX_FRAMES)
            ax1.set_ylim(0, 1)
            ax2.set_xlim(0, MAX_FRAMES)
            ax2.set_ylim(0, 1)
            
            # Redraw the plot
            plt.draw()
    except Exception as e:
        print(f"Error while updating plot: {e}")

# Set up the animation
ani = FuncAnimation(fig, update_plot, interval=100)

# Show the plot

plt.tight_layout()
plt.show()

# After the loop, close the serial connections
left.close()
right.close()

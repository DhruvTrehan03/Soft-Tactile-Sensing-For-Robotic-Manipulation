import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider

# Set up the grid
x = np.linspace(-10, 10, 30)
y = np.linspace(-10, 10, 30)
X, Y = np.meshgrid(x, y)

# Define charges (two sets of 5 points each)
charges = []
for y_val in [-2, -1, 0, 1, 2]:
    charges.append((-5, y_val, 1))  # Positive charges (1V)
    charges.append((5, y_val, -1))  # Negative charges (-1V)

def compute_electric_field(x_grid, y_grid, charges):
    Ex = np.zeros_like(x_grid)
    Ey = np.zeros_like(y_grid)
    epsilon = 1e-6  # To avoid division by zero
    for qx, qy, q in charges:
        dx = x_grid - qx
        dy = y_grid - qy
        r_squared = dx**2 + dy**2 + epsilon
        r_cubed = r_squared ** 1.5
        Ex += q * dx / r_cubed
        Ey += q * dy / r_cubed
    return Ex, Ey

# Calculate initial electric field
Ex, Ey = compute_electric_field(X, Y, charges)

# Set up figure and axis
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
plt.subplots_adjust(left=0.1, bottom=0.25)  # Make room for slider

# Plot electrodes
for qx, qy, q in charges:
    color = 'red' if q > 0 else 'blue'
    ax.scatter(qx, qy, c=color, s=100, marker='s', edgecolors='black')

# Quiver plot for electric field
quiver = ax.quiver(X, Y, Ex, Ey, scale=30, scale_units='inches', color='purple')

# Moving circle properties
radius = 2
circle = plt.Circle((0, 0), radius, edgecolor='black', facecolor='gray', alpha=0.5)
ax.add_patch(circle)

# Add conductivity slider
ax_cond = plt.axes([0.2, 0.1, 0.6, 0.03])
cond_slider = Slider(ax_cond, 'Conductivity', 0, 1, valinit=0.8)

def update(frame):
    # Update circle position (vertical oscillation)
    cy = 5 * np.sin(frame * 0.05)
    circle.center = (0, cy)
    
    # Get current conductivity value
    conductivity = cond_slider.val
    
    # Create mask for circle's area
    mask = (X - 0)**2 + (Y - cy)**2 <= radius**2
    
    # Modify electric field based on conductivity
    Ex_modified = np.where(mask, Ex * (1 - conductivity), Ex)
    Ey_modified = np.where(mask, Ey * (1 - conductivity), Ey)
    
    # Update quiver plot vectors
    quiver.set_UVC(Ex_modified, Ey_modified)
    
    return quiver, circle

# Create animation
ani = FuncAnimation(fig, update, frames=200, interval=50, blit=True)

plt.show()
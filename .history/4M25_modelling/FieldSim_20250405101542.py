import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Constants
k = 8.99e9  # Coulomb's constant (N·m²/C²)
q1 = 1e-6   # Charge of the first point charge (C)
q2 = -1e-6  # Charge of the second point charge (C)
r1 = np.array([-1.0, 0.0])  # Position of the first charge
r2 = np.array([1.0, 0.0])   # Position of the second charge
radius = 0.2  # Radius of the conducting ball

# Initialize the ball's position
ball_pos = np.array([0.0, 0.5])
ball_velocity = np.array([0.0, -0.02])  # Initial velocity of the ball

# Function to calculate the electric field at a point due to a charge
def electric_field(q, r_source, r_point):
    r = r_point - r_source
    r_magnitude = np.linalg.norm(r)
    if r_magnitude < 1e-6:  # Avoid division by zero
        return np.array([0.0, 0.0])
    return k * q * r / r_magnitude**3

# Function to calculate the net force on the ball
def net_force(ball_pos):
    # Image charges for the conducting ball
    image_charge1 = -q1 * radius / np.linalg.norm(ball_pos - r1)
    image_charge2 = -q2 * radius / np.linalg.norm(ball_pos - r2)
    
    # Positions of the image charges
    image_pos1 = r1 + (ball_pos - r1) * radius / np.linalg.norm(ball_pos - r1)
    image_pos2 = r2 + (ball_pos - r2) * radius / np.linalg.norm(ball_pos - r2)
    
    # Electric fields at the ball's position
    e1 = electric_field(q1, r1, ball_pos)
    e2 = electric_field(q2, r2, ball_pos)
    e_image1 = electric_field(image_charge1, image_pos1, ball_pos)
    e_image2 = electric_field(image_charge2, image_pos2, ball_pos)
    
    # Net electric field
    e_net = e1 + e2 + e_image1 + e_image2
    
    # Force on the ball (assuming it has a unit test charge)
    return e_net

# Function to calculate the electric field magnitude at a grid of points
def electric_field_magnitude(x, y):
    field = np.zeros_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            point = np.array([x[i, j], y[i, j]])
            e1 = electric_field(q1, r1, point)
            e2 = electric_field(q2, r2, point)
            field[i, j] = np.linalg.norm(e1 + e2)
    return field

# Set up the plot
fig, ax = plt.subplots()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_aspect('equal')
ax.set_title("Method of Image Charges Animation")

# Plot the stationary charges
charge1_plot, = ax.plot(r1[0], r1[1], 'ro', label="Charge 1 (+)")
charge2_plot, = ax.plot(r2[0], r2[1], 'bo', label="Charge 2 (-)")

# Plot the conducting ball
ball_plot, = ax.plot([], [], 'go', markersize=10, label="Conducting Ball")

# Create a grid of points
x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)

# Calculate the electric field magnitude at each grid point
field_magnitude = electric_field_magnitude(X, Y)

# Plot the electric field as a color map
field_plot = ax.contourf(X, Y, field_magnitude, levels=50, cmap='viridis', alpha=0.7)
cbar = fig.colorbar(field_plot, ax=ax)
cbar.set_label("Electric Field Magnitude")

# Update function for the animation
def update(frame):
    global ball_pos, ball_velocity
    
    # Calculate the net force on the ball
    force = net_force(ball_pos)
    
    # Update the ball's velocity and position (simple Euler integration)
    ball_velocity += force * 0.01  # Assume unit mass for simplicity
    ball_pos += ball_velocity
    
    # Reflect the ball if it hits the boundary
    if np.linalg.norm(ball_pos - r1) < radius or np.linalg.norm(ball_pos - r2) < radius:
        ball_velocity = -ball_velocity
    
    # Update the ball's position in the plot
    ball_plot.set_data([ball_pos[0]], [ball_pos[1]])  # Pass as sequences
    return ball_plot,

# Create the animation
ani = FuncAnimation(fig, update, frames=500, interval=20, blit=True)

# Add a legend
ax.legend()

# Show the animation
plt.show()
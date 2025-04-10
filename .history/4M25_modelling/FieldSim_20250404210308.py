import numpy as np
from matplotlib.animation import FuncAnimation

import matplotlib.pyplot as plt

# Constants
k = 8.99e9  # Coulomb's constant (N·m²/C²)
q1 = 1e-6   # Charge 1 (Coulombs)
q2 = -1e-6  # Charge 2 (Coulombs)
r1 = np.array([-1.0, 0.0])  # Position of charge 1
r2 = np.array([1.0, 0.0])   # Position of charge 2
ball_pos = np.array([0.0, -1.0])  # Start directly between the charges, below the midpoint
ball_velocity = np.array([0.0, 0.5])  # Initial velocity directed upwards

# Grid for field visualization
x = np.linspace(-2, 2, 50)
y = np.linspace(-2, 2, 50)
X, Y = np.meshgrid(x, y)

# Function to calculate electric field
def electric_field(q, r_source, x, y):
    r = np.sqrt((x - r_source[0])**2 + (y - r_source[1])**2)
    r_hat_x = (x - r_source[0]) / r
    r_hat_y = (y - r_source[1]) / r
    E_magnitude = k * q / r**2
    return E_magnitude * r_hat_x, E_magnitude * r_hat_y

# Calculate the total electric field
Ex1, Ey1 = electric_field(q1, r1, X, Y)
Ex2, Ey2 = electric_field(q2, r2, X, Y)
Ex = Ex1 + Ex2
Ey = Ey1 + Ey2

# Normalize the field for quiver plot
E_magnitude = np.sqrt(Ex**2 + Ey**2)
Ex /= E_magnitude
Ey /= E_magnitude

# Plot setup
fig, ax = plt.subplots()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_aspect('equal')
ax.set_title("Electric Field Simulation with Quiver Plot")
ax.set_xlabel("X")
ax.set_ylabel("Y")

# Plot charges
ax.scatter(*r1, color='red', s=100, label="Charge 1 (+)")
ax.scatter(*r2, color='blue', s=100, label="Charge 2 (-)")

# Plot electric field vectors using quiver
quiver = ax.quiver(X, Y, Ex, Ey, color='gray', scale=20, pivot='middle')

# Plot ball
ball, = ax.plot([], [], 'go', label="Ball")

# Update function for animation
def update(frame):
    global ball_pos, ball_velocity, stream
    # Calculate net force on the ball
    r_ball_to_q1 = ball_pos - r1
    r_ball_to_q2 = ball_pos - r2
    F1 = k * q1 / np.linalg.norm(r_ball_to_q1)**2 * r_ball_to_q1 / np.linalg.norm(r_ball_to_q1)
    F2 = k * q2 / np.linalg.norm(r_ball_to_q2)**2 * r_ball_to_q2 / np.linalg.norm(r_ball_to_q2)
    F_net = F1 + F2

    # Only consider the vertical component of the net force
    F_net[0] = 0  # Ignore horizontal force to keep the ball moving vertically

    # Update ball's velocity and position
    ball_velocity += F_net * 0.01  # Assume mass = 1 for simplicity
    ball_pos += ball_velocity * 0.01

    # Recalculate the electric field at each grid point
    Ex1, Ey1 = electric_field(q1, r1, X, Y)
    Ex2, Ey2 = electric_field(q2, r2, X, Y)
    Ex = Ex1 + Ex2
    Ey = Ey1 + Ey2

    # Normalize the field for streamplot
    E_magnitude = np.sqrt(Ex**2 + Ey**2)
    Ex_normalized = Ex / E_magnitude
    Ey_normalized = Ey / E_magnitude

    # Remove the previous streamlines
    for collection in ax.collections:
        collection.remove()

    # Update the streamplot
    stream = ax.streamplot(X, Y, Ex_normalized, Ey_normalized, color=np.log(E_magnitude), cmap='coolwarm', density=1.5)

    # Update ball's position in the plot
    ball.set_data([ball_pos[0]], [ball_pos[1]])
    return ball, stream.lines

# Create animation
ani = FuncAnimation(fig, update, frames=200, interval=50, blit=True)

# Show plot
plt.legend()
plt.show()
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parameters
dt = 0.001      # time step
t_max = 5       # total simulation time
steps = int(t_max/dt)
m = 1           # mass (arbitrary units)
alpha = 1       # polarizability

# Fixed charge positions and strength (unit charges)
x_left, x_right = -1.0, 1.0

def E_total(x):
    """Net electric field at position x from two charges."""
    # Avoid singularity: if x is too close to a charge, return a large number
    eps = 1e-6
    El = 1.0 / ( (x - x_left)**2 + eps )
    Er = -1.0 / ( (x - x_right)**2 + eps )
    return El + Er

def dE2_dx(x, dx=1e-6):
    """Numerical derivative of E_total(x)^2 with respect to x."""
    E2_forward = E_total(x + dx)**2
    E2_backward = E_total(x - dx)**2
    return (E2_forward - E2_backward) / (2*dx)

def force(x):
    """Compute force on the conducting ball at position x."""
    # F = (alpha/2) d/dx[E^2]
    return (alpha/2.0) * dE2_dx(x)

# Initial conditions: start at center with zero velocity
x = 0.0
v = 0.0

# Arrays to store simulation data
xs = []
ts = np.linspace(0, t_max, steps)

# Simple Euler-Cromer integration
for t in ts:
    xs.append(x)
    a = force(x) / m
    v = v + a*dt
    x = x + v*dt

xs = np.array(xs)

# Setup the figure and animation
fig, ax = plt.subplots(figsize=(8, 4))
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-1, 1)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Moving Conducting Ball between Two Charges')

# Plot fixed charges as red and blue dots
charge_size = 200
ax.scatter([x_left], [0], c='red', s=charge_size, label='+q')
ax.scatter([x_right], [0], c='blue', s=charge_size, label='-q')
ax.legend(loc='upper right')

# Draw the ball as a black circle (patch)
ball_radius = 0.1
ball = plt.Circle((xs[0], 0), ball_radius, color='black')
ax.add_patch(ball)

# Optionally, plot the electric field (for illustration)
x_field = np.linspace(-2.5, 2.5, 300)
E_field = np.array([E_total(xi) for xi in x_field])
# Scale the field for visualization purposes
E_scale = 0.2
ax.plot(x_field, E_field*E_scale, 'g--', label='E-field (scaled)')
ax.legend()

def animate(i):
    # Update ball position
    ball.center = (xs[i], 0)
    return ball,

ani = animation.FuncAnimation(fig, animate, frames=steps, interval=dt*1000, blit=True)

plt.show()

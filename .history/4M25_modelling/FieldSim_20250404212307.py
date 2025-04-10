import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

np.seterr(divide='ignore', invalid='ignore')

# grid size
N = 15
M = 25
# coordinates
X = np.arange(0, M, 1)
Y = np.arange(0, N, 1)
X, Y = np.meshgrid(X, Y)

# strength
Ex = np.zeros((N, M))
Ey = np.zeros((N, M))

# Define charges
charges = [
    (1, N // 2, M // 4),  # Fixed positive charge on the left
    (-1, N // 2, 3 * M // 4),  # Fixed negative charge on the right
    (1, 0, M // 2)  # Moving charge starts at the top middle
]

# Function to compute the electric field
def compute_field(charges):
    Ex = np.zeros((N, M))
    Ey = np.zeros((N, M))
    for q, qx, qy in charges:
        for i in range(N):
            for j in range(M):
                denom = ((i - qx) ** 2 + (j - qy) ** 2) ** 1.5
                if denom != 0: 
                    Ex[i, j] += q * (j - qy) / denom
                    Ey[i, j] += q * (i - qx) / denom
    return Ex, Ey

# Initialize the plot
fig, ax = plt.subplots(figsize=(12, 8))
field = ax.quiver(X, Y, Ex, Ey, pivot='mid')
charges_plot, = ax.plot([], [], 'bo')
cbar = plt.colorbar(field, ax=ax)
cbar.ax.set_ylabel('Magnitude')
ax.set_title('Electric Field Strength')
ax.axis('equal')
ax.axis('off')

# Update function for animation
def update(frame):
    global charges
    # Move the third charge up and down between the fixed charges
    q, _, qy = charges[2]
    qx = int(N / 2)
    qy = int(M / 2 + (M / 4) * np.sin(frame / 20))  # Oscillates vertically
    charges[2] = (q, qx, qy)

    # Recompute the field
    Ex, Ey = compute_field(charges)
    C = np.hypot(Ex, Ey)
    E = (Ex ** 2 + Ey ** 2) ** .5
    Ex = Ex / E
    Ey = Ey / E

    # Update the field and charges plot
    field.set_UVC(Ex, Ey, C)
    qq = [[], []]
    for _, qx, qy in charges:
        qq[0].append(qy)
        qq[1].append(qx)
    charges_plot.set_data(qq[0], qq[1])
    return field, charges_plot

# Create the animation
ani = FuncAnimation(fig, update, frames=200, interval=50, blit=True)

plt.show()
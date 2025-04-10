import numpy as np
import matplotlib.pyplot as plt
import random

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
# amount of charges
nq = 3

# computing
qq = [[], []]  # to store charges coordinates
for dummy in range(nq): 
    q = random.choice([-1, 1])
    qx, qy = random.randrange(1, N), random.randrange(1, M)
    # print(q, qx, qy)
    qq[0].append(qy)
    qq[1].append(qx)
    for i in range(N):
        for j in range(M):
            denom = ((i - qx) ** 2 + (j - qy) ** 2) ** 1.5
            if denom != 0: 
                Ex[i, j] += q * (j - qy) / denom
                Ey[i, j] += q * (i - qx) / denom

# arrows color
C = np.hypot(Ex, Ey)
# normalized values for arrows to be of equal length
E = (Ex ** 2 + Ey ** 2) ** .5
Ex = Ex / E
Ey = Ey / E

# drawing
plt.figure(figsize=(12, 8))
# charges
plt.plot(*qq, 'bo')
# field
plt.quiver(X, Y, Ex, Ey, C, pivot='mid')
# colorbar for magnitude
cbar = plt.colorbar()
cbar.ax.set_ylabel('Magnitude')
# misc
plt.title('Electric Field Strength')
plt.axis('equal')
plt.axis('off')
plt.show()
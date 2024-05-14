import numpy as np
# Nikon
ap = [[], [],[],[], [],[]]
ap[0] = [3e-4, 3e-4, 0.0011]
ap[1] = [8e-4, 0.0015, 0.0035]
ap[2] = [1e-4, 1e-4, 0]
ap[3] = [0,0,0]
ap[4] = [4e-4, 4e-4, 0.0022]
ap[5] = [2e-4, 7e-4, 6e-4]

for app in ap:
    mn = np.mean(app)
    std = np.std(app)
    print(str(np.round(mn,1)) + '\std{' + str(np.round(std,1)) + '}', end=' ')

# Sony
# means = 
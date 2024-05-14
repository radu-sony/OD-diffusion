import numpy as np
# sony
ap = [18.61, 18.65, 18.08]
ap50 = [33.57, 34.1, 33.71]
ap75 = [18.47, 18.84, 17.83]
apl = [41.31, 41.79, 58.75]
apm = [13.25, 15.14, 18.02]
aps = [1.84, 1.9, 1.9]

print('Sony')
print(f'& {np.round(np.mean(ap),1)}\std'+'{'+f'{np.round(np.std(ap),1)}'+'}', end=' ')
print(f'& {np.round(np.mean(ap50),1)}\std'+'{'+f'{np.round(np.std(ap50),1)}'+'}', end=' ')
print(f'& {np.round(np.mean(ap75),1)}\std'+'{'+f'{np.round(np.std(ap75),1)}'+'}', end=' ')
print(f'& {np.round(np.mean(aps),1)}\std'+'{'+f'{np.round(np.std(aps),1)}'+'}', end=' ')
print(f'& {np.round(np.mean(apm),1)}\std'+'{'+f'{np.round(np.std(apm),1)}'+'}', end=' ')
print(f'& {np.round(np.mean(apl),1)}\std'+'{'+f'{np.round(np.std(apl),1)}'+'}', end=' ')
print(' ')
#nikon
ap = [20.84, 22.46, 24.94]
ap50 = [36.61, 37,16, 42.12]
ap75 = [20.44, 22.37, 24.81]
apl = [50.06, 58.85, 58.75]
apm = [19.31, 17.08, 22.29]
aps = [1.99, 1.23, 1.9]

print('Nikon')
print(f'& {np.round(np.mean(ap),1)}\std'+'{'+f'{np.round(np.std(ap),1)}'+'}', end=' ')
print(f'& {np.round(np.mean(ap50),1)}\std'+'{'+f'{np.round(np.std(ap50),1)}'+'}', end=' ')
print(f'& {np.round(np.mean(ap75),1)}\std'+'{'+f'{np.round(np.std(ap75),1)}'+'}', end=' ')
print(f'& {np.round(np.mean(aps),1)}\std'+'{'+f'{np.round(np.std(aps),1)}'+'}', end=' ')
print(f'& {np.round(np.mean(apm),1)}\std'+'{'+f'{np.round(np.std(apm),1)}'+'}', end=' ')
print(f'& {np.round(np.mean(apl),1)}\std'+'{'+f'{np.round(np.std(apl),1)}'+'}', end=' ')
print(' ')

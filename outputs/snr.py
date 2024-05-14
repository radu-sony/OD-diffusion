import numpy as np

bit1 = 1/2**12
float32 = 1/2**32

scales = np.logspace(-4,0,10)

for scale in scales:
    snr = 20 * np.log10(bit1*scale / float32)
    print(np.round(scale,5), snr)
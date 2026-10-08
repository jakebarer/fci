import fci
import numpy as np
import scipy
dets = fci.make_dets()
H = fci.make_H(dets)
vals, vecs = np.linalg.eigh(H)
smallest_val = np.min(vals)
print(smallest_val)
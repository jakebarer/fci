import fci as f
from pyscf import gto, scf, ao2mo, fci
import numpy as np
import pyscf
h1e = np.load('h1e.npy')
h2e = np.load('h2e.npy')
dets = f.make_dets()
H = f.make_H(dets)
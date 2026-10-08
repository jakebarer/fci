import itertools
import numpy as np
# TODO make electron integrals global here
h1e = np.load('h1e.npy')
h2e = np.load('h2e.npy')
# swap from chemistry to phyiscs notation
# h2e = h2e.transpose(0,2,1,3)
# magic number(s)
basis_size = 6
# funcs for fci h6 sto3g specific, 6 spatial orbs 6 electrons

# def make_dets():
#     # spin up even, spin down odd
#     # 12 choose 6 dets
#     dets = itertools.combinations(range(12), 6)
#     return list(dets)
# def make_dets():
#     # only keep dets with exactly 3 alpha (even) and 3 beta (odd) electrons
#     dets = itertools.combinations(range(12), 6)
#     valid = []
#     for d in dets:
#         n_alpha = sum(1 for s in d if s % 2 == 0)
#         n_beta = sum(1 for s in d if s % 2 == 1)
#         if n_alpha == 3 and n_beta == 3:
#             valid.append(d)
#     return valid
def make_dets():
    # match pyscf for testing purposes
    norb = 6
    nelec = 3
    
    alpha_strings = list(itertools.combinations(range(norb), nelec))
    beta_strings  = list(itertools.combinations(range(norb), nelec))
    
    valid = []
    for a in alpha_strings:          # slow index (matches PySCF ci[I, J])
        for b in beta_strings:       # fast index
            # convert spatial orbitals to spin-orbitals
            # alpha orbital i → spin-orbital 2*i
            # beta  orbital i → spin-orbital 2*i + 1
            spinorb = tuple(sorted(
                [2*i for i in a] + [2*i + 1 for i in b]
            ))
            valid.append(spinorb)
    return valid
def spin_to_space(orb):
    # need space orbs for e- integrals so verbose func here
    return orb // 2
# need to check how much overlap between 2 individual dets
def det_overlap(d1, d2):
    # returns number of differences in spin orbitals, relevant for SC rules
    # compute 6 - |d1 intersect d2| (lists are size 6)
    intersect = list(set(d1) & set(d2))
    diffs = 6 - len(intersect)
    # get specific unique orbitals for SC rules later
    unique_d1 = [x for x in d1 if x not in d2]
    unique_d2 = [x for x in d2 if x not in d1]
    uniques = [unique_d1, unique_d2]
    return diffs, uniques

def make_H(dets):
    # am i hitting diagonal?
    n = len(dets)
    H = np.zeros((n,n))
    # fill in upper triangular then add transpose/fix diag
    # try brute force every index
    for j in range(n):
        for i in range(n):
            d1 = dets[i]
            d2 = dets[j]
            diffs, uniques = det_overlap(d1,d2)
            H[i,j] = SC_rule(diffs, uniques, d1, d2)
    # add in lower triangular
    # sym_H = H + H.T - np.diag(np.diag(H))
    sym_H = H
    return sym_H

def SC_rule(diffs, uniques, d1, d2):
    # do correct SC rule to eval electron integrals
    if diffs == 0:
        # add diagonal 1e integrals
        # make sure were adding the spatial orbs that actually are occupied in this version
        one_e = sum(h1e[spin_to_space(i), spin_to_space(i)] for i in d1)
        # add triangular 2e integrals
        two_e = 0.0
        for idx, i in enumerate(d1):
            for j in d1[idx+1:]:
                # check that im not integrating over same orbital: space(i) != space(j)
                # adding up triangular 2e ints of spatial orbs that are occupied here
                si, sj = spin_to_space(i), spin_to_space(j)
                # kill exchange term if opposite spins
                if i % 2 == j % 2:
                    two_e += (h2e[si,si,sj,sj] - h2e[si,sj,sj,si])
                else:
                    two_e += h2e[si,si,sj,sj]
        h = one_e + two_e
    elif diffs == 1:
        phase = get_phase(d1,d2)
        # 1 1e integral
        m_so = uniques[0][0] # det 1 unique spin
        p_so = uniques[1][0] # det 2 unique spin
        m = spin_to_space(m_so)
        p = spin_to_space(p_so)
        # spin orthogonality should kill this matrix element
        if m_so % 2 != p_so % 2:
            return 0.0
        one_e = h1e[m,p]
        # n 2e integrals
        two_e = 0.0
        # adding up 2e integrals for commonly occupied orbitals
        common = [x for x in d1 if x not in uniques[0]]
        for so in common:
            n = spin_to_space(so)
            two_e += h2e[m,p,n,n]
            # kill exchange term if opposite spins
            if m_so % 2 == so % 2:
                two_e -= h2e[m,n,n,p]
        h = phase * (one_e + two_e)
    elif diffs == 2:
        phase = get_phase(d1,d2)
        # 1 2e integral
        m_so = uniques[0][0]
        n_so = uniques[0][1]
        p_so = uniques[1][0]
        q_so = uniques[1][1]
        m = spin_to_space(m_so)
        n = spin_to_space(n_so)
        p = spin_to_space(p_so)
        q = spin_to_space(q_so)
        # kill exchange term if opposite spins
        coulomb = h2e[m,p,n,q] if m_so % 2 == p_so % 2 and n_so % 2 == q_so % 2 else 0.0
        exchange = h2e[m,q,n,p] if m_so % 2 == q_so % 2 and n_so % 2 == p_so % 2 else 0.0
        h = phase * (coulomb - exchange)
        # h = phase * (h2e[m,p,n,q] - h2e[m,q,n,p])
        # h = phase * h2e[m,p,n,q]
    else:
        h = 0.0
    return h

# def get_phase(d1, d2):
#     # counts number of det row exchanges required to get canonical det form and return phase sign accordingly
#     # n row changes, return (-1)**n
#     # first kill things in d1 not in d2, then create things unique in d2 and sort accordingly keeping track of idx
#     n = 0
#     d1copy = list(d1)
#     d2copy = list(d2)
#     d1uniques = [x for x in d1copy if x not in d2copy]
#     d2uniques = [x for x in d2copy if x not in d1copy]
#     # kill
#     for x in d1uniques:
#         idx = d1copy.index(x)
#         n += idx
#         d1copy.remove(x)
#     # create
#     for x in d2uniques:
#         # find how many are alr less than x since you have to move it that many times
#         idx = sum(1 for val in d1copy if val < x)
#         d1copy.insert(idx, x)
#         n += idx
#     return (-1)**n
# def get_phase(d1, d2):
#     n = 0
#     d1copy = list(d1)
#     d2copy = list(d2)
    
#     d1uniques = [x for x in d1copy if x not in d2copy]
#     d2uniques = [x for x in d2copy if x not in d1copy]
    
#     # Annihilate: count swaps to bring orbital to the left edge
#     for x in d1uniques:
#         idx = d1copy.index(x)
#         n += idx
#         d1copy.remove(x)
    
#     # Create: count swaps to insert from right edge to sorted position
#     for x in d2uniques:
#         idx = sum(1 for val in d1copy if val < x)
#         n += (len(d1copy) - idx)  # swaps from right, not left
#         d1copy.insert(idx, x)
    
#     return (-1)**n

def get_phase(d1,d2):
    d1copy = list(d1)
    d2copy = list(d2)
    d1uniques = [x for x in d1copy if x not in d2copy]
    d2uniques = [x for x in d2copy if x not in d1copy]
    if len(d1uniques) == 1:
        n = np.searchsorted(d1copy, d1uniques[0]) + np.searchsorted(d2copy, d2uniques[0])
    else:
        # len uniques should be 2, but only gets called for 1 and 2 case so this check should be good
        n = np.searchsorted(d1copy, d1uniques[0]) + np.searchsorted(d2copy, d2uniques[0]) + np.searchsorted(d1copy, d1uniques[1]) + np.searchsorted(d2copy, d2uniques[1])
    return (-1)**n

    
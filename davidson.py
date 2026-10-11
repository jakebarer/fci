import numpy as np
def proj(u,v):
    return (np.dot(v,u) / np.dot(u,u)) * u

def gram_schmidt(u, V):
    pr = 0.0
    # compute proj in each column
    for col in V.T:
        pr -= proj(u,col)
    return u - pr
def davidson(A):
    # start with random vec
    n = A.shape[0]
    v = np.rand(n)
    tol = 10e-8
    is_conv = False
    while not is_conv:
        # project
        Av = A@v
        vAv = v.conj().T @ Av
        # diagonalize in subspace
        mu, y = np.linalg.eig(vAv)
        # grab ritz pairs
        x = v@y
        lmbda = mu
        resid = A@x - lmbda*x
        if np.linalg.norm(resid) <= tol:
            return lmbda, x
        else:
            # precond is inverse diag(A)-lmbdaI
            # make orthonormal to v with gram schmidt and add it to that matrix
            M_inv = np.linalg.inv(np.diag(np.diag(A)) - np.diag(lmbda))
            w = M_inv @ resid
            w_orth = gram_schmidt(w, v)
            v = np.column_stack(v, w_orth)

import numpy as np
def proj(u,v):
    u = u.reshape(-1)
    v = v.reshape(-1)
    return (np.dot(v,u) / np.dot(u,u)) * u

def gram_schmidt(u, V):
    u = u.reshape(-1).copy()
    # compute proj in each column
    for col in V.T:
        u -= proj(col,u)
    return u / np.linalg.norm(u)
def davidson(A):
    # start with random vec
    n = A.shape[0]
    v = np.random.rand(n)
    v = v / np.linalg.norm(v)
    v = np.atleast_2d(v).T # force 2d array for first iteration
    tol = 1e-8
    while True:
        # project
        Av = A@v
        vAv = v.conj().T @ Av
        # diagonalize in subspace
        mu, y = np.linalg.eig(vAv)
        # grab lowest ritz pair
        idx = np.argmin(mu)
        lmbda = mu[idx]
        x = v@y[:, idx]
        resid = A@x - lmbda*x
        if np.linalg.norm(resid) <= tol:
            return lmbda, x
        else:
            # precond is inverse diag(A)-lmbdaI
            # make orthonormal to v with gram schmidt and add it to that matrix
            M_inv = np.linalg.inv(np.diag(np.diag(A)) - lmbda * np.eye((n)))
            w = M_inv @ resid
            w_orth = gram_schmidt(w, v)
            v = np.column_stack((v, w_orth))

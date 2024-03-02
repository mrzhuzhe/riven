import numpy as np

def arnoldi_iteration(A, b, n: int):
    """Compute a basis of the (n + 1)-Krylov subspace of the matrix A.

    This is the space spanned by the vectors {b, Ab, ..., A^n b}.

    Parameters
    ----------
    A : array_like
        An m × m array.
    b : array_like
        Initial vector (length m).
    n : int
        One less than the dimension of the Krylov subspace, or equivalently the *degree* of the Krylov space. Must be >= 1.
    
    Returns
    -------
    Q : numpy.array
        An m x (n + 1) array, where the columns are an orthonormal basis of the Krylov subspace.
    h : numpy.array
        An (n + 1) x n array. A on basis Q. It is upper Hessenberg.
    """
    eps = 1e-12
    h = np.zeros((n + 1, n))
    Q = np.zeros((A.shape[0], n + 1))
    # Normalize the input vector
    Q[:, 0] = b / np.linalg.norm(b, 2)  # Use it as the first Krylov vector
    for k in range(1, n + 1):
        v = np.dot(A, Q[:, k - 1])  # Generate a new candidate vector   
        print("v\n", v)     
        for j in range(k):  # Subtract the projections on previous vectors
            #h[j, k - 1] = np.dot(Q[:, j].conj(), v)
            h[j, k - 1] = np.dot(Q[:, j], v)
            v = v - h[j, k - 1] * Q[:, j]
        print("v\n", v)
        h[k, k - 1] = np.linalg.norm(v, 2)
        if h[k, k - 1] > eps:  # Add the produced vector to the list, unless
            Q[:, k] = v / h[k, k - 1]
        else:  # If that happens, stop iterating.
            return Q, h
    return Q, h


A = np.array([
    [2, -1, 0, 0],
    [-1, 2, -1, 0],    
    [0, -1, 2, -1],   
    [0, 0, -1, 2]
], dtype=np.double)

b = np.array([
    1, 2, 3, 4
], dtype=np.double)

if __name__ == "__main__":
    Q, h = arnoldi_iteration(A, b, 32)
    print(Q, "\n h: \n", h)
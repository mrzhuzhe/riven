import numpy as np

def householder(A):
    n = A.shape[0]
    v = np.zeros(n, dtype=np.double)
    u = np.zeros(n, dtype=np.double)
    z = np.zeros(n, dtype=np.double)

    for k in range(0, n - 2):

        if np.isclose(A[k + 1, k], 0.0):
            α = -np.sqrt(np.sum(A[(k + 1) :, k] ** 2))
        else:
            α = -np.sign(A[k + 1, k]) * np.sqrt(np.sum(A[(k + 1) :, k] ** 2))

        two_r_squared = α ** 2 - α * A[k + 1, k]
        v[k] = 0.0
        v[k + 1] = A[k + 1, k] - α
        v[(k + 2) :] = A[(k + 2) :, k]
        u[k:] = 1.0 / two_r_squared * np.dot(A[k:, (k + 1) :], v[(k + 1) :])
        z[k:] = u[k:] - np.dot(u, v) / (2.0 * two_r_squared) * v[k:]

        for l in range(k + 1, n - 1):

            A[(l + 1) :, l] = (
                A[(l + 1) :, l] - v[l] * z[(l + 1) :] - v[(l + 1) :] * z[l]
            )
            A[l, (l + 1) :] = A[(l + 1) :, l]
            A[l, l] = A[l, l] - 2 * v[l] * z[l]

        A[-1, -1] = A[-1, -1] - 2 * v[-1] * z[-1]
        A[k, (k + 2) :] = 0.0
        A[(k + 2) :, k] = 0.0

        A[k + 1, k] = A[k + 1, k] - v[k + 1] * z[k]
        A[k, k + 1] = A[k + 1, k]


A = np.array([
    [4, 1, -2, 2],
    [1, 2, 0, 1],    
    [-2, 0, 3, -2],   
    [2, 1, -2, -1]
], dtype=np.double)


if __name__ == "__main__":
    householder(A)
    print(A)
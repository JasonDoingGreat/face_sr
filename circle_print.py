import numpy as np


def circle_print(m, n):
    A = np.zeros(shape=(m, n))
    start_u = 0
    start_v = 0
    end_u = m
    end_v = n
    k = 0
    j = 0
    while start_u < end_u and start_u < end_v:
        for i in range(start_v, end_v):
            j += 1
            A[start_u, i] = j

        for i in range(start_u+1, end_u):
            j += 1
            A[i, end_v-1] = j

        if start_u != end_u:
            for i in range(end_v-2, start_v-1, -1):
                j += 1
                A[end_u-1, i] = j

        if start_v != end_v:
            for i in range(end_u-2, start_u, -1):
                j += 1
                A[i, start_v] = j

        k += 1
        start_u = k
        start_v = k
        end_u = m - k
        end_v = n - k

    return A


if __name__ == '__main__':
    result = circle_print(4, 4)
    print(result)

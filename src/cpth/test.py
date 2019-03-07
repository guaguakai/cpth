from cvxopt import solvers, matrix, spdiag, log
import numpy

if __name__ == "__main__":
    n = 2
    def F(x=None, z=None):
        if x is None: return 0, matrix(1.0, (n,1))
        if min(x) <= 0.0: return None
        f = x[0]**2 - x[0] * x[1] + x[1]**2
        Df = matrix([2*x[0] - x[1], -x[0] + 2*x[1]], (1,2))
        if z is None: return f, Df
        H = z[0] * matrix([[2, -1], [-1, 2]], (2,2), tc="d")
        return f, Df, H

    A = matrix([1,1], (1,2), tc="d")
    b = matrix([1], (1,1), tc="d")
    solution = solvers.cp(F, A=A, b=b)


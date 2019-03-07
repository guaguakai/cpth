import cvxopt
from cvxopt import matrix
import numpy as np

def forward_single_np(f, G_, h_, A_, b_):
    G, h, A, b = matrix(G_), matrix(h_), matrix(A_), matrix(b_)
    nz, neq, nineq = G.shape[1], A.shape[0], G.shape[0]

    solution = cvxopt.solvers.cp(f, G=G, h=h, A=A, b=b)
    print("status: {}".format(solution["status"]))
    obj, zhat, snl, sl, znl, zl, y = solution["primal_objective"], solution["x"], solution["snl"], solution["sl"], solution["znl"], solution["zl"], solution["y"]
    # x, slack of non-linear, slack of linear, dual variables of non-linear, dual variables of linear inequality, dual variables of the equality constraints

    return obj, zhat, snl, sl, znl, zl, y

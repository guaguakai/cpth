import cvxopt
import numpy
import torch
from torch.autograd import Variable
from torch.autograd import Function
from .util import extract_nBatch

class CPSolvers(Enum):
    PDIPM_BATCHED = 1
    CVXPY = 2
    GUROBI = 3
    CVXOPT = 4
    CUSTOM = 5

class CPFunction(Function):
    def __init__(self, eps=1e-12, verbose=0, notImprovedLim=3,
                 maxIter=20, solver=CPSolvers.GUROBI, model_params = None, custom_solver=None): # now only support gurobi
        self.eps = eps
        self.verbose = verbose
        self.notImprovedLim = notImprovedLim
        self.maxIter = maxIter
        self.solver = solver
        self.custom_solver = custom_solver
#        self.constant_constraints = constant_constraints

        if model_params is not None:
#        if constant_constraints:
#            self.A = A
#            self.b = b
#            self.G = G
#            self.h = h
#            A_arg = A.detach().numpy() if A is not None else None
#            b_arg = b.detach().numpy() if b is not None else None
#            G_arg = G.detach().numpy() if G is not None else None
#            h_arg = h.detach().numpy() if h is not None else None
#            model, x, inequality_constraints, equality_constraints, obj = make_gurobi_model(G_arg,
#                                                        h_arg, A_arg, b_arg)
            model, x, inequality_constraints, equality_constraints, obj = model_params
            self.model = model
            self.x = x
            self.inequality_constraints = inequality_constraints
            self.equality_constraints = equality_constraints
            self.quadobj = obj
        else:
            self.model = None


    # def forward(self, f_, fi_, hi_, G_, h_, A_, b_): # Not implemented yet
    def forward(self, f_, G_, h_, A_, b_):
        """ Solve a batch of CP
        The convex program can be formulated as:
            min f(z)
            s.t. f_i(z) <= 0 # not yet implemented
                 h_i(z) = 0 # not yet implemented
                 Gz <= h
                 Az = b

        """
        nBatch = extract_nBatch(torch.Tensor(), torch.Tensor(), G_, h_, A_, b_)
        G, _ = expandParam(G_, nBatch, 3)
        h, _ = expandParam(h_, nBatch, 2)
        A, _ = expandParam(A_, nBatch, 3)
        b, _ = expandParam(b_, nBatch, 2)

        _, nineq, nz = G.size()
        neq = A.size(1) if A.nelement() > 0 else 0
        assert(neq > 0 or nineq > 0)
        self.neq, self.nineq, self.nz = neq, nineq, nz

        if self.solver == QPSolvers.CVXOPT:
            Qs = torch.Tensor(nBatch, self.nz, self.nz).type_as(G_)  # the Hessian at the optimal solutions
            ps = torch.Tensor(nBatch, self.nz).type_as(G_)
            vals = torch.Tensor(nBatch).type_as(G_)                  # optimal objective value f_(z*)
            zhats = torch.Tensor(nBatch, self.nz).type_as(G_)        # optimal solution z*
            lams = torch.Tensor(nBatch, self.nineq).type_as(G_)      # dual variables of Gz <= h
            nus = torch.Tensor(nBatch, self.neq).type_as(G_)         # dual variables of Az = b
            slacks = torch.Tensor(nBatch, self.nineq).type_as(G_)    # slacks of Gz <= h, i.e., slacks = h - Gz

            for i in range(nBatch):
                vals[i], zhati, snli, sli, znli, zli, yi = solvers.cvxopt.forward_single_np(
                        f_, # fi_, hi_, # Not implemented yet
                        *[x.cpu().detach().numpy() if x is not None else None for x in [G_, h_, A_, b_]]
                        )
                zhats[i] = torch.Tensor(zhati)
                lams[i] = torch.Tensor(zli)
                slacks[i] = torch.Tensor(sli)
                if neq > 0:
                    nus[i] = torch.Tensor(yi)
                Qi = f_(zhati, cvxopt.matrix([[1.0]]))
                _, pi = f_(zhati)
                pi = pi - Qi * zhati
                Qs[i] = Qi
                ps[i] = pi

            self.vals = vals
            self.lams = lams
            self.nus = nus
            self.Qs = Qs
            self.ps = ps
            self.slacks = slacks

        else:
            print("Not yet implemented")
            assert False

        self.save_for_backward(zhats, G_, h_, A_, b_)
        return zhats

    def backward(self, dl_dzhat):
        zhats, G, h, A, b = self.saved_tensors
        Q = self.Qs
        p = self.ps
        G, G_e = expandParam(G, nBatch, 3)
        h, h_e = expandParam(h, nBatch, 2)
        A, A_e = expandParam(A, nBatch, 3)
        b, b_e = expandParam(b, nBatch, 2)

        nBatch = extract_nBatch(Q, p, G, h, A, b)

        if self.solver != QPSolvers.PDIPM_BATCHED:
            self.Q_LU, self.S_LU, self.R = pdipm_b.pre_factor_kkt(Q, G, A)

        d = torch.clamp(self.lams, min=1e-8) / torch.clamp(self.slacks, min=1e-8)

        pdipm_b.factor_kkt(self.S_LU, self.R, d)
        dx, _, dlam, dnu = pdipm_b.solve_kkt(
            self.Q_LU, d, G, A, self.S_LU,
            dl_dzhat, torch.zeros(nBatch, nineq).type_as(G),
            torch.zeros(nBatch, nineq).type_as(G),
            torch.zeros(nBatch, neq).type_as(G) if neq > 0 else torch.Tensor())

        # dps = dx
        dGs = bger(dlam, zhats) + bger(self.lams, dx)
        if G_e:
            dGs = dGs.mean(0).squeeze(0)
        dhs = -dlam
        if h_e:
            dhs = dhs.mean(0).squeeze(0)
        if neq > 0:
            dAs = bger(dnu, zhats) + bger(self.nus, dx)
            dbs = -dnu
            if A_e:
                dAs = dAs.mean(0).squeeze(0)
            if b_e:
                dbs = dbs.mean(0).squeeze(0)
        else:
            dAs, dbs = None, None
        dQs = 0.5 * (bger(dx, zhats) + bger(zhats, dx))
        if Q_e:
            dQs = dQs.mean(0).squeeze(0)
        # if p_e:
        #     dps = dps.mean(0).squeeze(0)

        grads = (dQs, dGs, dhs, dAs, dbs)

        return grads





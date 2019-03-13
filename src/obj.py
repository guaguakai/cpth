# import numpy as np
import autograd.numpy as np
# np.set_printoptions(threshold=np.nan)
import autograd

import scipy.optimize
import math
import gurobipy as gp
import argparse

import qpth
import torch
from torch.autograd import Variable, Function

from torch.utils.data import Dataset, DataLoader
from timeit import default_timer as timer
from sklearn.model_selection import train_test_split
import sklearn as skl
import sys
import pickle
import copy

from matching_utils import Net, load_data, make_matching_matrix

DTYPE = torch.float
DEVICE = torch.device("cpu")
visualization = False
verbose = True

class Dual(Function):
    def __init__(self, model, x_size, theta_size, m_size, edge_size):
        self.x_size = x_size
        self.theta_size = theta_size
        self.lamb_size = m_size
        self.m_size = m_size
        self.edge_size = edge_size
        self.Q = 1.0 * np.eye(self.x_size)
        self.P = 0.1 * np.eye(self.theta_size)
        self.model = model
        self.method = "SLSQP"
        # self.method = "Newton-CG"
        self.tol = 1e-3
        self.M = 1e3
        self.theta_bounds = [(-self.M,self.M)] * self.theta_size

    def m(self, theta, theta_bar, lib=np): # numpy inputs
        return (theta - theta_bar)
        # return (theta - theta_bar) ** 2 - 3

    def f(self, x, theta, lib=np): # default numpy inputs
        if lib == torch:
            Q = torch.Tensor(self.Q)
            P = torch.Tensor(self.P)
        else:
            Q = self.Q
            P = self.P
        return lib.dot(x, theta) + 0.5 * lib.dot(x, lib.matmul(Q, x)) - 0.5 * lib.dot(theta, lib.matmul(P, theta))

    # ============================ precompute derivatives =========================
    # entire_input: x, lamb, theta_bar, theta
    def m_single(self, entire_input):
        x = entire_input[:self.x_size]
        lamb = entire_input[self.x_size:self.x_size+self.m_size]
        theta_bar = entire_input[self.x_size + self.m_size : self.x_size + self.m_size + self.theta_size]
        theta = entire_input[-self.theta_size:]
        return np.dot(lamb, theta - theta_bar)

    def f_single(self, entire_input):
        x = entire_input[:self.x_size]
        lamb = entire_input[self.x_size:self.x_size+self.m_size]
        theta_bar = entire_input[self.x_size + self.m_size : self.x_size + self.m_size + self.theta_size]
        theta = entire_input[-self.theta_size:]
        return np.dot(x, theta) + 0.5 * np.transpose(x) @ self.Q @ x - 0.5 * np.transpose(theta) @ self.P @ theta

    def L_single(self, entire_input):
        x = entire_input[:self.x_size]
        lamb = entire_input[self.x_size:self.x_size+self.m_size]
        theta_bar = entire_input[self.x_size + self.m_size : self.x_size + self.m_size + self.theta_size]
        theta = entire_input[-self.theta_size:]
        L = -self.f_single(entire_input) + self.m_single(entire_input)
        return L

    def m_gradient_single(self, entire_input):
        return autograd.grad(self.m_single)(entire_input)

    def f_gradient_single(self, entire_input):
        return autograd.grad(self.f_single)(entire_input)

    def L_gradient_single(self, entire_input):
        return autograd.grad(self.L_single)(entire_input)

    def m_hessp_single(self, entire_input, p):
        m_gradientp = lambda x: np.dot(p, self.m_gradient_single(entire_input))
        return autograd.grad(m_gradientp)(entire_input)

    def f_hessp_single(self, entire_input, p):
        f_gradientp = lambda x: np.dot(p, self.f_gradient_single(entire_input))
        return autograd.grad(f_gradientp)(entire_input)

    def L_hessp_single(self, entire_input, p):
        L_gradientp = lambda x: np.dot(p, self.L_gradient_single(entire_input))
        return autograd.grad(L_gradientp)(entire_input)

    def m_hess_single(self, entire_input):
        return autograd.jacobian(self.m_gradient_single)(entire_input)

    def f_hess_single(self, entire_input):
        return autograd.jacobian(self.f_gradient_single)(entire_input)

    def L_hess_single(self, entire_input):
        return autograd.jacobian(self.L_gradient_single)(entire_input)


        

    # def L(self, x, lamb, theta, theta_bar, lib=np):
    #     value = -self.f(x, theta) + np.dot(self.m(theta, theta_bar), lamb)
        

class DualFunction(Dual):
    def forward(self, x_lambs, theta_bars):
        assert(x_lambs.dim() == 2 & theta_bars.dim() == 2)
        nBatch = len(x_lambs)
        obj_values = torch.Tensor(nBatch, 1).type_as(x_lambs)
        theta_values = torch.Tensor(nBatch, self.theta_size).type_as(x_lambs)
        jac_values = torch.Tensor(nBatch, self.theta_size).type_as(x_lambs)
        hessian_values = torch.Tensor(nBatch, self.theta_size, self.theta_size).type_as(x_lambs)
        for i in range(nBatch):
            x = x_lambs[i,:x_size].detach().numpy()
            lamb = x_lambs[i,x_size:].detach().numpy()
            theta_bar = theta_bars[i].detach().numpy()

            # ============= numpy scipy computing ===================
            # since numpy, torch, and autograd are not very consistent
            # we might need to transform everything back to numpy and finish computing the scipy part
            # then transform all of them back to torch in order to do the back propagation
            def lagrangian(theta):
                L = -self.f(x, theta) + np.dot(self.m(theta, theta_bar), lamb)
                return L

            lagrangian_jac = autograd.grad(lagrangian)
            lagrangian_hessian = autograd.jacobian(lagrangian_jac)
            # def lagrangian_jacp(theta, p):
            #     return np.dot(lagrangian_jac(theta), p)
            # lagrangian_hessianp = autograd.grad(lagrangian_jacp)
            res = scipy.optimize.minimize(fun=lagrangian, x0=theta_bar, method=self.method, jac=lagrangian_jac, tol=self.tol, bounds=self.theta_bounds) 

            obj_values[i] = torch.Tensor([res.fun])
            theta_values[i] = torch.Tensor(res.x)
            jac_values[i] = torch.Tensor(res.jac)
            hessian_values[i] = torch.Tensor(lagrangian_hessian(res.x))

        self.save_for_backward(x_lambs, theta_values, theta_bars, obj_values, jac_values, hessian_values)
        return obj_values

    def get_jac_torch(self, x_lambs, theta_bars, get_hess=False):
        assert(x_lambs.dim() == 2 & theta_bars.dim() == 2)
        nBatch = len(x_lambs)
        dg_dxlamb = torch.Tensor(*x_lambs.shape)
        hess_g = torch.Tensor(*x_lambs.shape, self.x_size + self.m_size)
        for i in range(nBatch):
            # ======================== same as forward path ============================
            # ------------------------- autograd version -------------------------------
            x = x_lambs[i,:x_size].detach().numpy()
            lamb = x_lambs[i,x_size:].detach().numpy() 
            theta_bar = theta_bars[i].detach().numpy() 

            def lagrangian(theta):
                L = -self.f(x, theta) + np.dot(self.m(theta, theta_bar), lamb)
                return L

            lagrangian_jac = autograd.grad(lagrangian)
            lagrangian_hessian = autograd.jacobian(lagrangian_jac)
            def lagrangian_jacp(theta, p):
                return np.dot(lagrangian_jac(theta), p)
            lagrangian_hessianp = autograd.grad(lagrangian_jacp)

            res = scipy.optimize.minimize(fun=lagrangian, x0=theta_bar, method=self.method, jac=lagrangian_jac, hessp=lagrangian_hessianp, tol=self.tol, bounds=self.theta_bounds) 

            # --------------------------- torch version --------------------------------
            # x = x_lambs[i,:x_size]
            # lamb = x_lambs[i,x_size:]
            # theta_bar = theta_bars[i]

            # def lagrangian(theta):
            #     theta_var = torch.Tensor(theta)
            #     L = -self.f(x, theta_var, lib=torch) + torch.dot(self.m(theta_var, theta_bar, lib=torch), lamb)
            #     return L.detach().numpy()

            # # lagrangian_jac = autograd.grad(lagrangian)
            # def lagrangian_jac(theta, get_hess=False):
            #     theta_var = Variable(torch.Tensor(theta), requires_grad=True)
            #     L = -self.f(x, theta_var, lib=torch) + torch.dot(self.m(theta_var, theta_bar, lib=torch), lamb)
            #     L_jac = torch.autograd.grad(L, theta_var)[0]
            #     return L_jac.detach().numpy()

            # # lagrangian_hessian = autograd.jacobian(lagrangian_jac)
            # def lagrangian_hessian(theta, get_hess=False):
            #     theta_var = Variable(torch.Tensor(theta), requires_grad=True)
            #     L_hess = torch.Tensor(theta.size, theta.size)
            #     L = -self.f(x, theta_var, lib=torch) + torch.dot(self.m(theta_var, theta_bar, lib=torch), lamb)
            #     L_jac = torch.autograd.grad(L, theta_var, retain_graph=True, create_graph=True)[0]
            #     for j in range(theta.size):
            #         L_hess[j] = torch.autograd.grad(L_jac[j], theta_var, retain_graph=True, create_graph=True)[0]

            #     return L_hess.detach().numpy()

            # res = scipy.optimize.minimize(fun=lagrangian, x0=theta_bar.detach().numpy(), method=self.method, jac=lagrangian_jac, hess=lagrangian_hessian) 

            # ========================== gradient computing =============================
            theta_torch = Variable(torch.Tensor(res.x), requires_grad=True)
            theta_bar_torch = Variable(theta_bars[i], requires_grad=True)
            x_lamb_torch = Variable(x_lambs[i], requires_grad=True)
            x_torch = x_lamb_torch[:x_size]
            lamb_torch = x_lamb_torch[x_size:]

            L = -self.f(x_torch, theta_torch, lib=torch) + torch.dot(self.m(theta_torch, theta_bar_torch, lib=torch), lamb_torch)
            L_jac = torch.autograd.grad(L, theta_torch, retain_graph=True, create_graph=True)[0]
            L_hess = torch.Tensor(self.theta_size, self.theta_size)
            for j in range(self.theta_size):
                L_hess[j] = torch.autograd.grad(L_jac[j], theta_torch, retain_graph=True, create_graph=True)[0]
            L_hess_inv = torch.inverse(L_hess)
            f_value = self.f(x_torch, theta_torch, lib=torch)
            df_dx = torch.autograd.grad(f_value, x_torch, retain_graph=True, create_graph=True)[0]
            df_dtheta = torch.autograd.grad(f_value, theta_torch, retain_graph=True, create_graph=True)[0]
            df_dthetadx = torch.Tensor(self.theta_size, self.x_size)
            for j in range(self.theta_size):
                df_dthetadx[j] = torch.autograd.grad(df_dtheta[j], x_torch, retain_graph=True, create_graph=True)[0]

            dtheta_dx = torch.matmul(L_hess_inv, df_dthetadx)

            m_value = self.m(theta_torch, theta_bar_torch, lib=torch)
            dm_dtheta = torch.Tensor(self.m_size, self.theta_size)
            for j in range(self.m_size):
                dm_dtheta[j] = torch.autograd.grad(m_value[j], theta_torch, retain_graph=True, create_graph=True)[0]

            dtheta_dlamb = torch.matmul(-L_hess_inv, dm_dtheta.transpose(-1,0))

            dg_dx = - df_dx - torch.matmul(df_dtheta, dtheta_dx) + torch.matmul(torch.matmul(lamb_torch, dm_dtheta), dtheta_dx)
            dg_dlamb = - torch.matmul(df_dtheta, dtheta_dlamb) + m_value + torch.matmul(torch.matmul(lamb_torch, dm_dtheta), dtheta_dlamb)
            dg_dxlamb[i] = torch.cat((dg_dx, dg_dlamb), dim=0)

            if get_hess:
                for j in range(self.x_size + self.m_size):
                    hess_g[i][j] = torch.autograd.grad(dg_dxlamb[i][j], x_lamb_torch, retain_graph=True, create_graph=True)[0]

        if get_hess:
            return dg_dxlamb, hess_g
        else:
            return dg_dxlamb

    def backward(self, dl_dg):
        x_lambs, thetas, theta_bars, obj_values, theta_jac, theta_hess = self.saved_tensors
        nBatch = len(x_lambs)
        dl_dxlamb = torch.Tensor(*x_lambs.shape)
        for i in range(nBatch):
            # ========================== gradient computing =============================
            theta_torch = Variable(torch.Tensor(thetas[i]), requires_grad=True)
            theta_bar_torch = Variable(torch.Tensor(theta_bars[i]), requires_grad=True)
            x_lamb_torch = Variable(torch.Tensor(x_lambs[i]), requires_grad=True)
            x_torch = x_lamb_torch[:x_size]
            lamb_torch = x_lamb_torch[x_size:]

            L = -self.f(x_torch, theta_torch, lib=torch) + torch.dot(self.m(theta_torch, theta_bar_torch, lib=torch), lamb_torch)
            L_jac = torch.autograd.grad(L, theta_torch, retain_graph=True, create_graph=True)[0]
            L_hess = torch.Tensor(self.theta_size, self.theta_size)
            for j in range(self.theta_size):
                L_hess[j] = torch.autograd.grad(L_jac[j], theta_torch, retain_graph=True, create_graph=True)[0]
            L_hess_inv = torch.inverse(L_hess)
            f_value = self.f(x_torch, theta_torch, lib=torch)
            df_dx = torch.autograd.grad(f_value, x_torch, retain_graph=True, create_graph=True)[0]
            df_dtheta = torch.autograd.grad(f_value, theta_torch, retain_graph=True, create_graph=True)[0]
            df_dthetadx = torch.Tensor(self.theta_size, self.x_size)
            for j in range(self.theta_size):
                df_dthetadx[j] = torch.autograd.grad(df_dtheta[j], x_torch, retain_graph=True, create_graph=True)[0]

            dtheta_dx = torch.matmul(L_hess_inv, df_dthetadx)

            m_value = self.m(theta_torch, theta_bar_torch, lib=torch)
            dm_dtheta = torch.Tensor(self.m_size, self.theta_size)
            for j in range(self.m_size):
                dm_dtheta[j] = torch.autograd.grad(m_value[j], theta_torch, retain_graph=True, create_graph=True)[0]

            dtheta_dlamb = torch.matmul(-L_hess_inv, dm_dtheta.transpose(-1,0))

            dg_dx = df_dx + torch.matmul(df_dtheta, dtheta_dx) - torch.matmul(torch.matmul(lamb_torch, dm_dtheta), dtheta_dx)
            dg_dlamb = torch.matmul(df_dtheta, dtheta_dlamb) - m_value - torch.matmul(torch.matmul(lamb_torch, dm_dtheta), dtheta_dlamb)
            dg_dxlamb = torch.cat((dg_dx, dg_dlamb), dim=0)
            dl_dxlamb[i] = torch.matmul(dl_dg, dg_dxlamb)
        return dl_dxlamb, None # TODO

class DualGradient(Dual):
    def forward(self, x_lambs, theta_bars):
        import torch
        from torch.autograd import Variable

        assert(x_lambs.dim() == 2 & theta_bars.dim() == 2)
        nBatch = len(x_lambs)
        dg_dxlamb = torch.Tensor(*x_lambs.shape)
        for i in range(nBatch):
            # ======================== same as forward path ============================
            # ------------------------- autograd version -------------------------------
            x = x_lambs[i,:x_size].detach().numpy()
            lamb = x_lambs[i,x_size:].detach().numpy() 
            theta_bar = theta_bars[i].detach().numpy() 

            def lagrangian(theta):
                L = -self.f(x, theta) + np.dot(self.m(theta, theta_bar), lamb)
                return L

            lagrangian_jac = autograd.grad(lagrangian)
            lagrangian_hessian = autograd.jacobian(lagrangian_jac)
            def lagrangian_jacp(theta, p):
                return np.dot(lagrangian_jac(theta), p)
            lagrangian_hessianp = autograd.grad(lagrangian_jacp)

            res = scipy.optimize.minimize(fun=lagrangian, x0=theta_bar, method=self.method, jac=lagrangian_jac, hessp=lagrangian_hessianp, tol=self.tol, bounds=self.theta_bounds, constraints=()) 

            # ========================== gradient computing =============================
            theta = res.x
            entire_input = np.concatenate((x, lamb, theta_bar, theta))

            L = self.L_single(entire_input)
            L_jac = self.L_gradient_single(entire_input)
            L_hess = self.L_hess_single(entire_input)

            L_hess_theta = L_hess[-self.theta_size:,-self.theta_size:]
            # L_hess_theta_inv = np.linalg.inv(L_hess_theta) # theta_size by theta_size
            # dtheta_dx = - L_hess_theta_inv @ L_hess[-self.theta_size:, :-self.theta_size] # TODO: this could be done by using GUROBI or CPLEX
            dtheta_dx = - np.linalg.solve(L_hess_theta, L_hess[-self.theta_size:, :-self.theta_size])

            dentire_dx = np.concatenate((np.eye(self.x_size + self.lamb_size + self.theta_size), dtheta_dx), axis=0)
            dg_dx = L_jac @ dentire_dx
            print(dg_dx)
            dg_dxlamb[i] = torch.Tensor(dg_dx[:-self.theta_size]) # without the last gradient of theta_bar
            # TODO...

        return dg_dxlamb

    def backward(self, dl_dg):
        x_lambs, thetas, theta_bars, obj_values, theta_jac, theta_hess = self.save_tensors
        nBatch = len(x_lambs)
        dl_dxlamb = torch.Tensor(*x_lambs.shape)
        for i in range(nBatch):
            print("TODO")
            # ========================== gradient computing =============================
        return dl_dxlamb, None # TODO


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Matching')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                        help='input batch size for testing (default: 1)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--truncated-size', type=int, default=10, metavar='N',
                        help='how many nodes in each side of the bipartite graph')

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)

    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

    # =============================================================================

    edge_size = args.truncated_size**2
    A, b = make_matching_matrix(args.truncated_size)

    x_size     = edge_size 
    theta_size = edge_size 
    m_size     = edge_size 
    lamb_size  = m_size
    M = 1e3
    tol = 1e-3

    train_loader, test_loader = load_data(args, kwargs)

    # def m(theta, y):
    #     batch_size = len(y)
    #     theta_bar = model(y)
    #     theta_bar = torch.reshape(theta_bar, (batch_size, edge_size))
    #     print("YOLO")
    #     return theta - theta_bar

    # def f(x, theta):
    #     # x = x_theta[:x_size]
    #     # theta = x_theta[x_size:]
    #     return x.transpose(-1,0) @ theta + 0.5 * x.transpose(-1,0) @ Q @ x - 0.5 * theta.transpose(-1,0) @ P @ theta

    model = Net().to(DEVICE)
    dual_function = DualFunction(model=model, x_size=x_size, theta_size=theta_size, m_size=m_size, edge_size=edge_size)
    dual_gradient = DualGradient(model=model, x_size=x_size, theta_size=theta_size, m_size=m_size, edge_size=edge_size)

    nBatch = args.batch_size
    print(nBatch)
    x = 1.0 * torch.ones((nBatch, x_size)) # TODO wrong dimension
    lamb = 0.1 * torch.ones((nBatch, lamb_size)) # TODO wrong dimension

    for batch_idx, (features, labels) in enumerate(train_loader):
        features, labels = features.to(DEVICE), labels.to(DEVICE)
        theta_bars = model(features).view(nBatch, theta_size)
        x_lamb = torch.cat((x,lamb), dim=1)
        obj_value = dual_function(x_lamb, theta_bars)
        # obj_value, theta_opt, theta_jac, theta_hessian = obj_function.value(x, lamb, features)
        # g_jac, g_hess = dual_function.get_jac_torch(x_lamb, theta_bar, get_hess=True)
        break

    theta_bar = theta_bars[0:1,:]

    def g(x):
        x_torch = torch.Tensor(x).view(1, x_size + lamb_size)
        value = -dual_function(x_torch, theta_bar).detach().numpy()[0]
        print(value)
        return value

    def g_jac(x):
        x_torch = torch.Tensor(x).view(1, x_size + lamb_size)
        gradient = -dual_gradient(x_torch, theta_bar).detach().numpy()[0]
        # gradient = -dual_function.get_jac_torch(x_torch, theta_bar).detach().numpy()[0]
        return gradient

    def g_hess(x):
        x_torch = torch.Tensor(x).view(1, x_size + lamb_size)
        gradient, hessian = dual_function.get_jac_torch(x_torch, theta_bar, get_hess=True)
        return -hessian.detach().numpy()[0]


    Q = torch.Tensor(dual_function.Q)
    P = torch.Tensor(dual_function.P)
    x_opt = - torch.matmul(torch.inverse(Q), theta_bar.view(-1))
    lamb_opt = torch.matmul(torch.eye(lamb_size) + torch.matmul(P, Q), x_opt)
    xlamb_opt = torch.cat((x_opt, lamb_opt), dim=0).view(1,2*edge_size).detach().numpy()
    xlamb_opt_torch = Variable(torch.cat((x_opt, lamb_opt), dim=0).view(1,2*edge_size), requires_grad=True)

    gradient = dual_gradient(xlamb_opt_torch, theta_bar)


    def eq_fun(x):
        return A @ x[:x_size] - b
    def budget_fun(x):
        return - np.sum(x[:x_size]) + 10 

    constraints_slsqp = []
    # constraints_slsqp.append(scipy.optimize.LinearConstraint(A, b, b))
    constraints_slsqp.append({"type": "eq", "fun": eq_fun, "jac": autograd.jacobian(budget_fun)})
    # constraints_slsqp.append(scipy.optimize.LinearConstraint(np.ones((1, x_size)), np.array([-np.inf]), np.array([10])))
    constraints_slsqp.append({"type": "ineq", "fun": budget_fun, "jac": autograd.grad(budget_fun)})

    print("minimizing...")
    res = scipy.optimize.minimize(fun=g, x0=0.5 * np.ones((1,2*edge_size)), method="SLSQP", jac=g_jac, bounds=[(-M, M)]*(edge_size) + [(0.0, M)]*(edge_size), constraints=constraints_slsqp, tol=tol)
    print(res)


    

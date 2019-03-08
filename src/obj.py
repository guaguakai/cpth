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

from matching_utils import Net, load_data

DTYPE = torch.float
DEVICE = torch.device("cpu")
visualization = False
verbose = True

class DualFunction(Function):
    def __init__(self, model, x_size, theta_size, m_size):
        self.x_size = x_size
        self.theta_size = theta_size
        self.edge_size = 400
        self.Q = 0.5 * np.eye(self.x_size)
        self.P = 0.2 * np.eye(self.theta_size)
        self.m_size = m_size
        self.model = model
        self.method = "SLSQP"

    def m(self, theta, theta_bar, lib=np): # numpy inputs
        print("YOLO")
        if lib == torch:
            tmp_theta_bar = torch.Tensor(theta_bar)
            return theta - tmp_theta_bar.view(self.theta_size)
        else:
            tmp_theta_bar = theta_bar
            return theta - lib.reshape(tmp_theta_bar, (self.theta_size))

    def f(self, x, theta, lib=np): # default numpy inputs
        if lib == torch:
            Q = torch.Tensor(self.Q)
            P = torch.Tensor(self.P)
        else:
            Q = self.Q
            P = self.P
        return lib.dot(x, theta) + 0.5 * lib.dot(x, lib.matmul(Q, x)) - 0.5 * lib.dot(theta, lib.matmul(P, theta))

    def forward(self, x_lambs, theta_bars):
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
                return -self.f(x, theta) + np.dot(self.m(theta, theta_bar), lamb)

            lagrangian_jac = autograd.grad(lagrangian)
            lagrangian_hessian = autograd.jacobian(lagrangian_jac)
            res = scipy.optimize.minimize(fun=lagrangian, x0=np.zeros((self.theta_size)), method=self.method, jac=lagrangian_jac, hess=lagrangian_hessian) 

            obj_values[i] = torch.Tensor([-res.fun])
            theta_values[i] = torch.Tensor(res.x)
            jac_values[i] = torch.Tensor(res.jac)
            hessian_values[i] = torch.Tensor(lagrangian_hessian(res.x))

        self.save_for_backward(x_lambs, theta_values, theta_bars, obj_values, jac_values, hessian_values)
        return obj_values

    def get_jac_torch(self, x_lambs, theta_bars):
        nBatch = len(x_lambs)
        dg_dxlamb = torch.Tensor(*x_lambs.shape)
        hess_g = torch.Tensor(*x_lambs.shape, self.x_size + self.m_size)
        for i in range(nBatch):
            # ======================== same as forward path ============================
            x = x_lambs[i,:x_size].detach().numpy()
            lamb = x_lambs[i,x_size:].detach().numpy()
            theta_bar = theta_bars[i].detach().numpy()

            def lagrangian(theta):
                return -self.f(x, theta) + np.dot(self.m(theta, theta_bar), lamb)

            lagrangian_jac = autograd.grad(lagrangian)
            lagrangian_hessian = autograd.jacobian(lagrangian_jac)
            res = scipy.optimize.minimize(fun=lagrangian, x0=np.zeros((self.theta_size)), method=self.method, jac=lagrangian_jac, hess=lagrangian_hessian) 

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

            dg_dx = df_dx + torch.matmul(df_dtheta, dtheta_dx) - torch.matmul(torch.matmul(lamb_torch, dm_dtheta), dtheta_dx)
            dg_dlamb = torch.matmul(df_dtheta, dtheta_dlamb) - m_value - torch.matmul(torch.matmul(lamb_torch, dm_dtheta), dtheta_dlamb)
            dg_dxlamb[i] = torch.cat((dg_dx, dg_dlamb), dim=0)

            for j in range(self.x_size + self.m_size):
                hess_g[i][j] = torch.autograd.grad(dg_dxlamb[i][j], x_lamb_torch, retain_graph=True, create_graph=True)[0]

        return dg_dxlamb, hess_g

    def backward(self, dl_dg):
        x_lambs, thetas, theta_bars, obj_values, theta_jac, theta_hess = self.save_tensors
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


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Matching')
    parser.add_argument('--batch-size', type=int, default=2, metavar='N',
                        help='input batch size for training (default: 2)')
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
    parser.add_argument('--truncated-size', type=int, default=20, metavar='N',
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

    x_size     = edge_size 
    theta_size = edge_size 
    m_size     = edge_size 
    lamb_size  = m_size

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
    dual_function = DualFunction(model=model, x_size=x_size, theta_size=theta_size, m_size=m_size)

    nBatch = args.batch_size
    print(nBatch)
    x = 1 * torch.ones((nBatch, x_size)) # TODO wrong dimension
    lamb = 0.1 * torch.ones((nBatch, lamb_size)) # TODO wrong dimension

    for batch_idx, (features, labels) in enumerate(train_loader):
        features, labels = features.to(DEVICE), labels.to(DEVICE)
        theta_bar = model(features)
        x_lamb = torch.cat((x,lamb), dim=1)
        obj_value = dual_function(x_lamb, theta_bar)
        # obj_value, theta_opt, theta_jac, theta_hessian = obj_function.value(x, lamb, features)
        print(obj_value)
        g_jac, g_hess = dual_function.get_jac_torch(x_lamb, theta_bar)
        break
    

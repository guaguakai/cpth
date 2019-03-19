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
from torch.autograd import Function

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
    def __init__(self, model, x_size, theta_size, m_size, edge_size, phi_size):
        self.x_size = x_size
        self.theta_size = theta_size
        self.lamb_size = m_size
        self.m_size = m_size
        self.phi_size = phi_size
        self.edge_size = edge_size
        self.Q = 0.1 * np.eye(self.x_size)
        self.P = 0.05 * np.eye(self.theta_size)
        self.model = model
        self.method = "SLSQP"
        # self.method = "Newton-CG"
        self.tol = 1e-3
        self.M = 1e3
        self.theta_bounds = [(-self.M,self.M)] * self.theta_size
        self.constraint_matrix = np.concatenate((np.eye(self.x_size), -np.eye(self.x_size)), axis=0)

    def m(self, theta, phi, lib=np): # numpy inputs
        theta_bar = phi[:self.theta_size]
        r = phi[-self.theta_size:]
        return self.constraint_matrix @ (theta - theta_bar) - np.concatenate((r,r), axis=0)
        # return (theta - phi) ** 2 - 3

    def f(self, x, theta, lib=np): # default numpy inputs
        if lib == torch:
            Q = torch.Tensor(self.Q)
            P = torch.Tensor(self.P)
        else:
            Q = self.Q
            P = self.P
        return lib.dot(x, theta) + 0.5 * lib.dot(x, lib.matmul(Q, x)) - 0.5 * lib.dot(theta, lib.matmul(P, theta))

    def dual_solution(self, x, lamb, phi):
        theta_bar = phi[:self.x_size]
        theta = np.linalg.solve(self.P, x - np.transpose(self.constraint_matrix) @ lamb)
        return theta

    # ================ Lagrangian and gradient computing =================
    # entire input: x, lamb, phi, theta
    def L_single(self, entire_input):
        x = entire_input[:self.x_size]
        lamb = entire_input[self.x_size : self.x_size + self.lamb_size]
        phi = entire_input[self.x_size + self.lamb_size : self.x_size + self.lamb_size + self.phi_size]
        theta = entire_input[-self.theta_size:]
        L = -self.f(x, theta) + np.dot(lamb, self.m(theta, phi))
        return L

    def L_gradient_single(self, entire_input):
        return autograd.grad(self.L_single)(entire_input)

    def L_gradient_single_direct(self, entire_input):
        x = entire_input[:self.x_size]
        lamb = entire_input[self.x_size : self.x_size + self.lamb_size]
        phi = entire_input[self.x_size + self.lamb_size : self.x_size + self.lamb_size + self.phi_size]
        theta = entire_input[-self.theta_size:]
        dL_dx = -theta -self.Q @ x
        dL_dlamb = self.m(theta, phi)
        dL_dphi = np.concatenate((np.transpose(self.constraint_matrix), -(np.concatenate((np.eye(self.x_size), np.eye(self.x_size)), axis=1))), axis=0) @ lamb # TODO error!!
        dL_dtheta = -x + self.P @ theta + np.transpose(self.constraint_matrix) @ lamb
        return np.concatenate((dL_dx, dL_dlamb, dL_dphi, dL_dtheta), axis=0)

    def L_hessp_single(self, entire_input, p):
        L_gradientp = lambda x: np.dot(p, self.L_gradient_single_direct(entire_input))
        return autograd.grad(L_gradientp)(entire_input)

    def L_hess_single(self, entire_input):
        return autograd.jacobian(self.L_gradient_single_direct)(entire_input)

        

class DualFunction(Dual):
    def forward(self, x_lambs, phis):
        assert(x_lambs.dim() == 2 & phis.dim() == 2)
        nBatch = len(x_lambs)
        obj_values = torch.Tensor(nBatch, 1).type_as(x_lambs)
        theta_values = torch.Tensor(nBatch, self.theta_size).type_as(x_lambs)
        for i in range(nBatch):
            x = x_lambs[i,:self.x_size].detach().numpy()
            lamb = x_lambs[i,self.x_size:].detach().numpy()
            phi = phis[i].detach().numpy()

            theta = self.dual_solution(x, lamb, phi)
            fun = self.L_single(np.concatenate((x, lamb, phi, theta)))

            obj_values[i] = torch.Tensor([fun])
            theta_values[i] = torch.Tensor(theta)

        # self.save_for_backward(x_lambs, theta_values, phis, obj_values)
        return obj_values

    def backward(self, dl_dg):
        # ***This function might not be used
        return None, None # TODO

class DualGradient(Dual):
    def forward(self, x_lambs, phis):
        import torch
        from torch.autograd import Variable

        assert(x_lambs.dim() == 2 & phis.dim() == 2)
        nBatch = len(x_lambs)
        dg_dx = torch.Tensor(nBatch, self.x_size + self.lamb_size + self.phi_size)
        theta_values = torch.Tensor(nBatch, self.theta_size).type_as(x_lambs)
        jac_values = torch.Tensor(nBatch, self.x_size + self.lamb_size + self.theta_size + self.theta_size).type_as(x_lambs)
        hessian_values = torch.Tensor(nBatch, self.theta_size, self.theta_size).type_as(x_lambs)
        for i in range(nBatch):
            # ======================== same as forward path ============================
            x = x_lambs[i,:self.x_size].detach().numpy()
            lamb = x_lambs[i,self.x_size:].detach().numpy() 
            phi = phis[i].detach().numpy() 

            theta = self.dual_solution(x, lamb, phi)
            entire_input = np.concatenate((x, lamb, phi, theta))

            L = self.L_single(entire_input)
            L_jac = self.L_gradient_single_direct(entire_input)
            L_hess = self.L_hess_single(entire_input)

            L_hess_theta = L_hess[-self.theta_size:,-self.theta_size:]
            dtheta_dx = - np.linalg.solve(L_hess_theta, L_hess[-self.theta_size:, :-self.theta_size])

            dentire_dx = np.concatenate((np.eye(self.x_size + self.lamb_size + self.phi_size), dtheta_dx), axis=0)
            dg_dx[i] = torch.Tensor(L_jac @ dentire_dx)
            # dg_dxlamb[i] = torch.Tensor(dg_dx[:-self.theta_size]) # without the last gradient of phi
            # TODO...
            theta_values[i] = torch.Tensor(theta)
            
        self.save_for_backward(x_lambs, theta_values, phis, jac_values, hessian_values)
        return dg_dx

    def backward(self, dl_dg):
        x_lambs, thetas, phis, theta_jac, theta_hess = self.saved_tensors
        nBatch = len(x_lambs)
        dl_dxlamb = torch.Tensor(*x_lambs.shape)
        dl_dphis = torch.Tensor(*phis.shape)
        for i in range(nBatch):
            x = x_lambs[i,:self.x_size].detach().numpy()
            lamb = x_lambs[i,self.x_size:].detach().numpy() 
            phi = phis[i].detach().numpy() 
            theta = thetas[i].detach().numpy()

            entire_input = np.concatenate((x, lamb, phi))

            def g_gradient(entire_without_theta):
                entire = np.concatenate((entire_without_theta, theta))
                L = self.L_single(entire)
                L_jac = self.L_gradient_single_direct(entire)
                L_hess = self.L_hess_single(entire)

                L_hess_theta = L_hess[-self.theta_size:,-self.theta_size:]
                dtheta_dx = - np.linalg.solve(L_hess_theta, L_hess[-self.theta_size:, :-self.theta_size])

                dentire_dx = np.concatenate((np.eye(self.x_size + self.lamb_size + self.theta_size), dtheta_dx), axis=0)
                gradientp = np.dot(dl_dg, (L_jac @ dentire_dx))
                return gradientp

            hessp = torch.Tensor(autograd.grad(g_gradient)(entire_input))
            dl_dxlamb[i] = hessp[:self.x_size + self.lamb_size]
            dl_dphis[i] = hessp[-self.theta_size:]

            # ========================== gradient computing =============================
        return dl_dxlamb, dl_dphis # TODO

class DualHess(Dual):
    def hess(self, x_lambs, phis):
        nBatch, x_lamb_size = x_lambs.shape
        hess = torch.Tensor(nBatch, x_lamb_size, x_lamb_size)
        for i in range(nBatch):
            x = x_lambs[i,:self.x_size].detach().numpy()
            lamb = x_lambs[i,self.x_size:].detach().numpy() 
            phi = phis[i].detach().numpy() 
            theta = self.dual_solution(x, lamb, phi)

            entire_input = np.concatenate((x, lamb, phi))

            def g_gradient(entire_without_theta):
                entire = np.concatenate((entire_without_theta, theta))
                L_jac = self.L_gradient_single_direct(entire)
                L_hess = self.L_hess_single(entire)

                L_hess_theta = L_hess[-self.theta_size:,-self.theta_size:]
                dtheta_dx = - np.linalg.solve(L_hess_theta, L_hess[-self.theta_size:, :-self.theta_size])
                dentire_dx = np.concatenate((np.eye(self.x_size + self.lamb_size + self.phi_size), dtheta_dx), axis=0)
                gradient = (L_jac @ dentire_dx)[:self.x_size + self.lamb_size]
                return gradient

            hess[i] = torch.Tensor(autograd.jacobian(g_gradient)(entire_input))[:,self.x_size + self.lamb_size]
        return hess

    def hessp(self, x_lambs, phis, p):
        assert(x_lambs.dim() == 2 & phis.dim() == 2)
        nBatch = len(x_lambs)
        hessp_g = torch.Tensor(*x_lambs.shape)
        for i in range(nBatch):
            x = x_lambs[i,:self.x_size].detach().numpy()
            lamb = x_lambs[i,self.x_size:].detach().numpy() 
            phi = phis[i].detach().numpy() 
            theta = self.dual_solution(x, lamb, phi)

            entire_input = np.concatenate((x, lamb, phi))

            def g_gradientp(entire_without_theta):
                entire = np.concatenate((entire_without_theta, theta))
                L_jac = self.L_gradient_single_direct(entire)
                L_hess = self.L_hess_single(entire)

                L_hess_theta = L_hess[-self.theta_size:,-self.theta_size:]
                dtheta_dx = - np.linalg.solve(L_hess_theta, L_hess[-self.theta_size:, :-self.theta_size])
                dentire_dx = np.concatenate((np.eye(self.x_size + self.lamb_size + self.phi_size), dtheta_dx), axis=0)
                gradientp = np.dot(p, (L_jac @ dentire_dx)[:self.x_size + self.lamb_size])
                return gradientp

            hessp = torch.Tensor(autograd.grad(g_gradientp)(entire_input))
            hessp_g[i] = hessp[:self.x_size + self.lamb_size]

        return hessp_g



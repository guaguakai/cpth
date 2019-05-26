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

# from matching_utils import Net, load_data, make_matching_matrix

DTYPE = torch.float
DEVICE = torch.device("cpu")
visualization = False
verbose = True

class Dual():
    def __init__(self, x_size, theta_size, m_size, edge_size, phi_size, constraint_matrix):
        self.x_size = x_size
        self.theta_size = theta_size
        self.lamb_size = m_size
        self.m_size = m_size
        self.phi_size = phi_size
        self.edge_size = edge_size
        self.Q = 0.01 * np.eye(self.x_size)     # The larger the Q and P, the more relaxation effect to the final obj. Of course, it is also more stable.
        self.P = 0.01 * np.eye(self.theta_size)
        self.P_inv = np.linalg.inv(self.P)
        self.method = "SLSQP"
        # self.method = "Newton-CG"
        self.tol = 1e-4
        # self.M = 1e3
        # self.theta_bounds = [(-self.M,self.M)] * self.theta_size
        # self.constraint_matrix = 1.0 / self.x_size * np.random.normal(size=(self.m_size, self.x_size)) # random constraints
        # self.constraint_matrix = 1.0 / self.x_size * np.ones((1, self.x_size)) # budget constraint
        # self.constraint_matrix = np.concatenate((np.eye(self.x_size), -np.eye(self.x_size)), axis=0) # box constraints
        # self.constraint_matrix = np.concatenate((np.eye(self.x_size), -np.eye(self.x_size), 1.0 / self.x_size * np.random.normal(size=(self.m_size - 2 * self.x_size, self.x_size))), axis=0) # box constraints with random constraints
        # self.constraint_matrix = np.concatenate((np.eye(self.x_size), -np.eye(self.x_size), 0.1 * np.array([[0, 1, 0, 0, 1]])), axis=0) # box constraints with random constraints
        self.constraint_matrix_np = np.array(constraint_matrix)
        self.constraint_matrix = torch.Tensor(constraint_matrix)

        assert(self.m_size == self.constraint_matrix.shape[0])

        self.Q_extended = np.concatenate(
                (-np.concatenate((self.Q + self.P_inv, self.P_inv @ self.constraint_matrix_np.transpose()), axis=1),
                np.concatenate((self.constraint_matrix_np @ self.P_inv, -self.constraint_matrix_np @ self.P_inv @ self.constraint_matrix_np.transpose()), axis=1)),
                axis=0)

    def m(self, theta, phi, lib=np): # numpy inputs
        theta_bar = phi[:self.theta_size]
        r = phi[self.theta_size:]
        if lib == np:
            return self.constraint_matrix_np @ (theta - theta_bar) - r
        elif lib == torch:
            return self.constraint_matrix @ (theta - theta_bar) - r
        # return (theta - phi) ** 2 - 3

    def dm_dphi(self, theta, phi, lib=np):
        if lib == np:
            return -lib.concatenate((self.constraint_matrix_np, lib.eye(self.phi_size - self.theta_size)), axis=1)
        elif lib == torch:
            return -lib.cat((self.constraint_matrix, lib.eye(self.phi_size - self.theta_size)), dim=1)

    def f(self, x, theta, lib=np): # default numpy inputs
        if lib == torch:
            Q = torch.Tensor(self.Q)
            P = torch.Tensor(self.P)
        else:
            Q = self.Q
            P = self.P
        return lib.dot(x, theta) + 0.5 * lib.dot(x, lib.matmul(Q, x)) - 0.5 * lib.dot(theta, lib.matmul(P, theta))

    def dual_solution(self, x, lamb, phi, lib=np):
        # theta_bar = phi[:self.x_size]
        if lib == np:
            theta = self.P_inv @ (x - np.transpose(self.constraint_matrix_np) @ lamb)
        elif lib == torch:
            theta = torch.Tensor(self.P_inv) @ (x - self.constraint_matrix.t() @ lamb)
        return theta

    def dtheta_dx(self, lib=np):
        if lib == np:
            P_inv_t = np.transpose(self.P_inv)
            return np.concatenate((P_inv_t, - P_inv_t @ np.transpose(self.constraint_matrix_np), np.zeros((self.theta_size, self.phi_size))), axis=1) # TODO to check
        elif lib == torch:
            P_inv_t = torch.Tensor(self.P_inv).t()
            return torch.cat((P_inv_t, - P_inv_t @ self.constraint_matrix.t(), torch.zeros((self.theta_size, self.phi_size))), dim=1)

    # ================ Lagrangian and gradient computing =================
    # entire input: x, lamb, phi, theta
    def L_single(self, entire_input):
        x = entire_input[:self.x_size]
        lamb = entire_input[self.x_size : self.x_size + self.lamb_size]
        phi = entire_input[self.x_size + self.lamb_size : self.x_size + self.lamb_size + self.phi_size]
        theta = entire_input[-self.theta_size:]
        L = -self.f(x, theta) + np.dot(lamb, self.m(theta, phi))
        return L

    # def L_gradient_single(self, entire_input):
    #     return autograd.grad(self.L_single)(entire_input)

    # def L_gradient_single_direct(self, entire_input, lib=np):
    #     x = entire_input[:self.x_size]
    #     lamb = entire_input[self.x_size : self.x_size + self.lamb_size]
    #     phi = entire_input[self.x_size + self.lamb_size : self.x_size + self.lamb_size + self.phi_size]
    #     theta = entire_input[-self.theta_size:]
    #     dL_dx = -theta -self.Q @ x
    #     dL_dlamb = self.m(theta, phi)
    #     # dL_dphi = np.concatenate((np.transpose(self.constraint_matrix_np), -(np.concatenate((np.eye(self.x_size), np.eye(self.x_size)), axis=1))), axis=0) @ lamb # TODO error!!
    #     dL_dphi = np.concatenate((np.transpose(self.constraint_matrix_np) @ lamb, -lamb), axis=0) # TODO modified on 5/23, to be checked
    #     dL_dtheta = -x + self.P @ theta + np.transpose(self.constraint_matrix_np) @ lamb
    #     return np.concatenate((dL_dx, dL_dlamb, dL_dphi, dL_dtheta), axis=0)

    # def L_hessp_single(self, entire_input, p):
    #     L_gradientp = lambda x: np.dot(p, self.L_gradient_single_direct(entire_input))
    #     return autograd.grad(L_gradientp)(entire_input)

    # def L_hess_single(self, entire_input):
    #     return autograd.jacobian(self.L_gradient_single_direct)(entire_input)

    # def L_theta_hess(self, entire_input):
    #     x = entire_input[:self.x_size]
    #     lamb = entire_input[self.x_size : self.x_size + self.lamb_size]
    #     phi = entire_input[self.x_size + self.lamb_size : self.x_size + self.lamb_size + self.phi_size]
    #     theta = entire_input[-self.theta_size:]

    #     dL_dtheta = -x + self.P @ theta + np.transpose(self.constraint_matrix_np) @ lamb
    #     hess = self.P
    #     return hess

    def g_gradient_torch(self, entire_input, phi, lib=np):
        x = entire_input[:self.x_size]
        lamb = entire_input[self.x_size : self.x_size + self.lamb_size]
        # phi = entire_input[self.x_size + self.lamb_size : self.x_size + self.lamb_size + self.phi_size]
        theta = self.dual_solution(x, lamb, phi, lib=lib)

        if lib == np:
            constraint_matrix = self.constraint_matrix_np
            dtheta_dx = self.dtheta_dx(lib=lib)
            dL_dx = -theta - self.Q @ x
            dL_dlamb = self.m(theta, phi, lib=lib)
            dL_dphi = np.transpose(self.dm_dphi(theta, phi, lib=lib)) @ lamb # TODO error!!
            dL_dtheta = -x + self.P @ theta + np.transpose(constraint_matrix) @ lamb

            dg_dx = np.concatenate((dL_dx, dL_dlamb, dL_dphi)) + dL_dtheta @ dtheta_dx

        elif lib == torch:
            constraint_matrix = self.constraint_matrix
            dtheta_dx = torch.Tensor(self.dtheta_dx(lib=lib))
            dL_dx = -theta - torch.Tensor(self.Q) @ x
            dL_dlamb = self.m(theta, phi, lib=torch)
            dL_dphi = self.dm_dphi(theta, phi, lib=lib).t() @ lamb # TODO error!!
            dL_dtheta = -x + torch.Tensor(self.P) @ theta + constraint_matrix.t() @ lamb

            dg_dx = torch.cat((dL_dx, dL_dlamb, dL_dphi)) + dL_dtheta @ dtheta_dx
        return dg_dx

        
        

class DualFunction(Dual):
    def __call__(self, x_lambs, phis):
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

        return obj_values



class DualGradient(Dual):
    def __call__(self, x_lambs, phis):
        assert(x_lambs.dim() == 2 & phis.dim() == 2)
        nBatch = len(x_lambs)
        dg_dx = torch.Tensor(nBatch, self.x_size + self.lamb_size + self.phi_size)
        theta_values = torch.Tensor(nBatch, self.theta_size).type_as(x_lambs)
        for i in range(nBatch):
            # ======================== same as forward path ============================
            x_lamb = x_lambs[i]
            phi = phis[i] 

            dg_dx[i] = self.g_gradient_torch(x_lamb, phi, lib=torch)
            # dg_dxlamb[i] = torch.Tensor(dg_dx[:-self.theta_size]) # without the last gradient of phi
            # TODO...
            
        return dg_dx

    # def backward(self, dl_dg):
    #     x_lambs, thetas, phis = self.saved_tensors
    #     nBatch = len(x_lambs)
    #     dl_dxlamb = torch.Tensor(*x_lambs.shape)
    #     dl_dphis = torch.Tensor(*phis.shape)
    #     for i in range(nBatch):
    #         x = x_lambs[i,:self.x_size].detach().numpy()
    #         lamb = x_lambs[i,self.x_size:].detach().numpy() 
    #         phi = phis[i].detach().numpy() 
    #         theta = thetas[i].detach().numpy()

    #         entire_input = np.concatenate((x, lamb, phi))

    #         def g_gradient(entire_without_theta):
    #             entire = np.concatenate((entire_without_theta, theta))
    #             L_jac = self.L_gradient_single_direct(entire)
    #             L_hess = self.L_hess_single(entire)

    #             L_hess_theta = L_hess[-self.theta_size:,-self.theta_size:]
    #             # dtheta_dx = - np.linalg.solve(L_hess_theta, L_hess[-self.theta_size:, :-self.theta_size])

    #             L_hess_theta_inv = np.linalg.inv(L_hess_theta)
    #             dtheta_dx = - L_hess_theta_inv @ L_hess[-self.theta_size:, :-self.theta_size]

    #             dentire_dx = np.concatenate((np.eye(self.x_size + self.lamb_size + self.theta_size), dtheta_dx), axis=0)
    #             gradientp = np.dot(dl_dg, (L_jac @ dentire_dx))
    #             return gradientp

    #         hessp = torch.Tensor(autograd.grad(g_gradient)(entire_input))
    #         dl_dxlamb[i] = hessp[:self.x_size + self.lamb_size]
    #         dl_dphis[i] = hessp[-self.theta_size:]

    #         # ========================== gradient computing =============================
    #     return dl_dxlamb, dl_dphis # TODO

class DualHess(Dual):
    def hess(self, x_lambs, phis):
        nBatch, x_lamb_size = x_lambs.shape
        hess = torch.Tensor(nBatch, x_lamb_size, x_lamb_size)
        for i in range(nBatch):
            x_lamb = x_lambs[i].detach().numpy()
            phi = phis[i].detach().numpy() 

            # def g_gradient(entire_without_theta):
            #     print("autograd starts...")
            #     print(entire_without_theta)
            #     entire = np.concatenate((entire_without_theta, theta))
            #     print(entire)
            #     L_jac = self.L_gradient_single_direct(entire)
            #     print(L_jac)
            #     L_hess = self.L_hess_single(entire)
            #     print("L hessian")
            #     print(L_hess)

            #     L_hess_theta = L_hess[-self.theta_size:,-self.theta_size:]
            #     # dtheta_dx = - np.linalg.solve(L_hess_theta, L_hess[-self.theta_size:, :-self.theta_size])

            #     L_hess_theta_inv = np.linalg.inv(L_hess_theta)
            #     print(L_hess_theta_inv)
            #     dtheta_dx = - L_hess_theta_inv @ L_hess[-self.theta_size:, :-self.theta_size]

            #     dentire_dx = np.concatenate((np.eye(self.x_size + self.lamb_size + self.phi_size), dtheta_dx), axis=0)
            #     gradient = (L_jac @ dentire_dx)[:self.x_size + self.lamb_size]
            #     print("autograd finishes...")
            #     return gradient

            # hess[i] = torch.Tensor(autograd.jacobian(g_gradient)(entire_input))[:,self.x_size + self.lamb_size]

            hess[i] = torch.Tensor(autograd.jacobian(self.g_gradient_torch)(x_lamb, phi)[:self.x_size + self.lamb_size, : self.x_size + self.lamb_size])
            # hess[i] = torch.Tensor(self.Q_extended)
        return hess

    def hessp(self, x_lambs, phis, p):
        assert(x_lambs.dim() == 2 & phis.dim() == 2)
        nBatch = len(x_lambs)
        hessp_g = torch.Tensor(*x_lambs.shape)
        for i in range(nBatch):
            x_lamb = x_lambs[i].detach().numpy()
            phi = phis[i].detach().numpy() 

            # def g_gradientp(entire_without_theta):
            #     entire = np.concatenate((entire_without_theta, theta))
            #     L_jac = self.L_gradient_single_direct(entire)
            #     L_hess = self.L_hess_single(entire)

            #     L_hess_theta = L_hess[-self.theta_size:,-self.theta_size:]
            #     dtheta_dx = - np.linalg.solve(L_hess_theta, L_hess[-self.theta_size:, :-self.theta_size])
            #     dentire_dx = np.concatenate((np.eye(self.x_size + self.lamb_size + self.phi_size), dtheta_dx), axis=0)
            #     gradientp = np.dot(p, (L_jac @ dentire_dx)[:self.x_size + self.lamb_size])
            #     return gradientp

            def g_gradientp(x_lamb, phi):
                # x_lamb = entire[:self.x_size + self.lamb_size]
                # phi = entire[-self.phi_size:]
                gradient = self.g_gradient_torch(x_lamb, phi)[:self.x_size + self.lamb_size]
                return np.dot(p, gradient)

            hessp = torch.Tensor(autograd.grad(g_gradientp)(x_lamb, phi))
            hessp_g[i] = hessp[:self.x_size + self.lamb_size]

        return hessp_g



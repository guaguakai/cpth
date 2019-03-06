import autograd.numpy as np
# np.set_printoptions(threshold=np.nan)
# from autograd import grad
# from autograd import jacobian

import scipy.optimize
import math
import gurobipy as gp

import qpth
import torch
from torch.autograd import Variable

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

class objectiveFunction():
    def __init__(self, f, m, model, x_size, theta_size, m_size):
        self.x_size = x_size
        self.theta_size = theta_size
        self.m_size = m_size
        self.model = model

    # def m(self, theta, y):
    #     theta_bar = model(y)
    #     return theta - theta_bar

    # def f(self, x_theta):
    #     x = x_theta[:x_size]
    #     theta = x_theta[x_size:]
    #     return np.transpose(x) @ theta + 0.5 * np.transpose(x) @ Q @ x - 0.5 * np.transpose(theta) @ P @ theta

    def value(self, x_, lamb_, y):
        x = Variable(x_, requires_grad=True)
        lamb = Variable(lamb_, requires_grad=True)
        lagrangian = lambda theta: -f(x, theta) + np.sum(m(theta, y) * lamb) # TODO
        lagrangian_jac = grad(lagrangian)
        lagrangian_hessian = jacobian(lagrangian_jac)
        res = scipy.optimize.minimize(fun=lagrangian, x0=np.zeros(self.theta_size), method="SLSQP", jac=lagrangian_jac, hess=lagrangian_hessian)

        theta_opt = res.x
        theta_jac = res.jac
        theta_hessian = res.hess
        obj_value = res.fun

        return obj_value, theta_opt, theta_jac, theta_hessian


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
    parser.add_argument('--truncated-size', type=int, default=50, metavar='N',
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

    Q = 0.5 * np.eye(5)
    P = 0.2 * np.eye(5)
    x_size = 5
    theta_size = 5
    m_size = 5

    train_loader, test_loader = load_data(args, kwargs)


    def m(theta, y):
        theta_bar = model(y)
        return theta - theta_bar

    def f(x, theta):
        # x = x_theta[:x_size]
        # theta = x_theta[x_size:]
        return np.transpose(x) @ theta + 0.5 * np.transpose(x) @ Q @ x - 0.5 * np.transpose(theta) @ P @ theta

    obj_function = objectiveFunction(f=f, m=m, model=Net().to(DEVICE), x_size=x_size, theta_size=theta_size, m_size=m_size)

    jac = grad(f)
    x = np.array([1.0, 1.0, 1.0, 1.0, 0])
    theta = np.array([1.0, 1.0, 0, 0, 1.0])
    print(jac(np.concatenate([x,theta])))
    hessian = jacobian(jac)(np.concatenate([x,theta]))
    print(hessian)

    obj_value, theta_opt, theta_jac, theta_hessian = obj_function.value(np.concatenate([x,theta]))
    

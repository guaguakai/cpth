# import numpy as np
import autograd.numpy as np
# np.set_printoptions(threshold=np.nan)
import autograd

import qpth
import torch
from torch.autograd import Variable, Function
import scipy.optimize
import argparse

from torch.utils.data import Dataset, DataLoader
from timeit import default_timer as timer
from sklearn.model_selection import train_test_split
import sklearn as skl
import sys
import pickle
import copy
import time

from matching_utils import Net, load_data, make_matching_matrix
from toy_obj import Dual, DualFunction, DualGradient, DualHess

DTYPE = torch.float
DEVICE = torch.device("cpu")
visualization = False
verbose = True


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
    G, h = make_matching_matrix(args.truncated_size)
    A, b = torch.Tensor(), torch.Tensor()

    x_size     = edge_size
    theta_size = edge_size
    m_size     = edge_size * 2
    lamb_size  = m_size
    phi_size = edge_size * 2
    M = 1e3
    tol = 1e-3
    # method = "SLSQP"
    method = "trust-constr"

    train_loader, test_loader = load_data(args, kwargs)

    # def m(theta, y):
    #     batch_size = len(y)
    #     phi = model(y)
    #     phi = torch.reshape(theta_bar, (batch_size, edge_size))
    #     print("YOLO")
    #     return theta - phi

    # def f(x, theta):
    #     # x = x_theta[:x_size]
    #     # theta = x_theta[x_size:]
    #     return x.transpose(-1,0) @ theta + 0.5 * x.transpose(-1,0) @ Q @ x - 0.5 * theta.transpose(-1,0) @ P @ theta

    model = Net().to(DEVICE)
    uncertainty_model = Net().to(DEVICE)
    dual_function = DualFunction(model=model, x_size=x_size, theta_size=theta_size, m_size=m_size, edge_size=edge_size, phi_size=phi_size)
    dual_gradient = DualGradient(model=model, x_size=x_size, theta_size=theta_size, m_size=m_size, edge_size=edge_size, phi_size=phi_size)
    dual_hess = DualHess(model=model, x_size=x_size, theta_size=theta_size, m_size=m_size, edge_size=edge_size, phi_size=phi_size)

    nBatch = args.batch_size
    print(nBatch)
    x = 1.0 * torch.ones((nBatch, x_size)) # TODO wrong dimension
    lamb = 0.1 * torch.ones((nBatch, lamb_size)) # TODO wrong dimension

    for batch_idx, (features, labels) in enumerate(train_loader):
        features, labels = features.to(DEVICE), labels.to(DEVICE)
        mean = model(features).view(nBatch, theta_size)
        variance = uncertainty_model(features).view(nBatch, theta_size)
        phis = torch.cat((mean, variance), dim=1)
        x_lamb = torch.cat((x,lamb), dim=1)
        obj_value = dual_function(x_lamb, phis)
        # obj_value, theta_opt, theta_jac, theta_hessian = obj_function.value(x, lamb, features)
        # g_jac, g_hess = dual_function.get_jac_torch(x_lamb, phi, get_hess=True)
        break

    phi = phis[0:1,:]

    def g(x):
        x_torch = torch.Tensor(x).view(1, x_size + lamb_size)
        value = -dual_function(x_torch, phi).detach().numpy()[0]
        print(value)
        return value

    def g_jac(x):
        x_torch = torch.Tensor(x).view(1, x_size + lamb_size)
        gradient = -dual_gradient(x_torch, phi).detach().numpy()[0][:x_size + lamb_size]
        # gradient = -dual_function.get_jac_torch(x_torch, phi).detach().numpy()[0]
        return gradient

    def g_hess(x):
        x_torch = torch.Tensor(x).view(1, x_size + lamb_size)
        gradient, hessian = dual_hess.hess(x_torch, phi, get_hess=True)
        return -hessian.detach().numpy()[0]

    def g_hessp(x, p):
        x_torch = torch.Tensor(x).view(1, x_size + lamb_size)
        p_torch = torch.Tensor(p)
        hessp = dual_hess.hessp(x_torch, phi, p_torch)
        return -hessp.detach().numpy()[0]

    def ineq_fun(x):
        return G @ x[:x_size] - h
    def budget_fun(x):
        return np.array([- np.sum(x[:x_size]) + 10])

    start_time = time.time()
    constraints_slsqp = []
    # constraints_slsqp.append(scipy.optimize.LinearConstraint(A, b, b))
    constraints_slsqp.append({"type": "ineq", "fun": ineq_fun, "jac": autograd.jacobian(ineq_fun)})
    # constraints_slsqp.append(scipy.optimize.LinearConstraint(np.ones((1, x_size)), np.array([-np.inf]), np.array([10])))
    # constraints_slsqp.append({"type": "ineq", "fun": budget_fun, "jac": autograd.grad(budget_fun)})

    print("minimizing...")
    res = scipy.optimize.minimize(fun=g, x0=0.5 * np.ones((x_size + lamb_size)), method=method, jac=g_jac, hessp=g_hessp, bounds=[(0.0, M)]*(x_size) + [(0.0, M)]*(lamb_size), constraints=constraints_slsqp)
    print(res)

    xlamb_torch = Variable(torch.ones(x_size + lamb_size).view(1, x_size + lamb_size), requires_grad=True)
    gradient = dual_gradient(xlamb_torch, phi)[0]
    test = torch.dot(gradient[:x_size + lamb_size], torch.ones(x_size + lamb_size))
    # grad_of_grad = torch.autograd.grad(test, xlamb_torch)[0]
    # print(grad_of_grad)
    print(g(xlamb_torch))
    print(g_jac(xlamb_torch))
    p = torch.ones(x_size + lamb_size)
    print(g_hessp(xlamb_torch, p))

    print("running time: {}".format(time.time() - start_time))

    # newG = torch.nn.functional.pad(G, (0, lamb_size, 0, lamb_size), "constant", 0)
    # newG[-lamb_size:, -lamb_size:] = -torch.eye(lamb_size)
    # newh = torch.nn.functional.pad(h, (0, lamb_size), "constant", 0)
    # newA = torch.nn.functional.pad(A, (0,0,0,lamb_size), "constant", 0)
    # newb = torch.nn.functional.pad(b, (0, lamb_size), "constant", 0)






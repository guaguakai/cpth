# import numpy as np
import autograd.numpy as np
# np.set_printoptions(threshold=np.nan)
import autograd

import qpth
import qpthlocal
import torch
import torch.optim as optim
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
import tqdm

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
    # A, b = torch.Tensor(), torch.Tensor()

    x_size     = edge_size
    theta_size = edge_size
    m_size     = edge_size * 2
    lamb_size  = m_size
    phi_size = edge_size * 2
    M = 1e3
    tol = 1e-3
    method = "SLSQP"
    # method = "trust-constr"

    train_loader, test_loader = load_data(args, kwargs)

    model = Net().to(DEVICE)
    uncertainty_model = Net().to(DEVICE)
    dual_function = DualFunction(model=model, x_size=x_size, theta_size=theta_size, m_size=m_size, edge_size=edge_size, phi_size=phi_size)
    dual_gradient = DualGradient(model=model, x_size=x_size, theta_size=theta_size, m_size=m_size, edge_size=edge_size, phi_size=phi_size)
    dual_hess = DualHess(model=model, x_size=x_size, theta_size=theta_size, m_size=m_size, edge_size=edge_size, phi_size=phi_size)

    nBatch = args.batch_size
    print(nBatch)
    x = 1.0 * torch.ones((nBatch, x_size)) # TODO wrong dimension
    lamb = 0.1 * torch.ones((nBatch, lamb_size)) # TODO wrong dimension

    def ineq_fun(x):
        return G @ x[:x_size] - h

    constraints_slsqp = []
    # constraints_slsqp.append(scipy.optimize.LinearConstraint(A, b, b))
    constraints_slsqp.append({"type": "ineq", "fun": ineq_fun, "jac": autograd.jacobian(ineq_fun)})

    learning_rate = 1e-2
    num_epochs = 50
    optimizer = optim.SGD(list(model.parameters()) + list(uncertainty_model.parameters()), lr=learning_rate, momentum=0.5)

    for epoch in tqdm.trange(num_epochs):
        training_loss = []
        # ======================= training ==========================
        for batch_idx, (features, labels) in enumerate(train_loader):
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            mean = model(features).view(nBatch, theta_size)
            variance = uncertainty_model(features).view(nBatch, theta_size)
            phis = torch.cat((-mean, variance), dim=1)
            x_lamb = torch.cat((x,lamb), dim=1)
            obj_value = dual_function(x_lamb, phis)

            def g(x):
                x_torch = torch.Tensor(x).view(1, x_size + lamb_size)
                value = -dual_function(x_torch, phis).detach().numpy()[0]
                # print(value)
                return value

            def g_jac(x):
                x_torch = torch.Tensor(x).view(1, x_size + lamb_size)
                gradient = -dual_gradient(x_torch, phis).detach().numpy()[0][:x_size + lamb_size]
                # gradient = -dual_function.get_jac_torch(x_torch, phi).detach().numpy()[0]
                return gradient

            def g_hess(x):
                x_torch = torch.Tensor(x).view(1, x_size + lamb_size)
                hess = dual_hess.hess(x_torch, phis)
                return -hess.detach().numpy()[0]

            def g_hessp(x, p):
                x_torch = torch.Tensor(x).view(1, x_size + lamb_size)
                p_torch = torch.Tensor(p)
                hessp = dual_hess.hessp(x_torch, phis, p_torch)
                return -hessp.detach().numpy()[0]

            start_time = time.time()
            res = scipy.optimize.minimize(fun=g, x0= np.random.rand((x_size + lamb_size)), method=method, jac=g_jac, hessp=g_hessp, bounds=[(-M, M)]*(x_size) + [(0.0, M)]*(lamb_size), constraints=constraints_slsqp, options={"maxiter": 10})

            xlamb = torch.Tensor(res.x).view(1, x_size + lamb_size)

            newG = np.pad(G, ((0, lamb_size), (0, lamb_size)), "constant", constant_values=0)
            newG[-lamb_size:, -lamb_size:] = -torch.eye(lamb_size)
            newh = np.pad(h, (0, lamb_size), "constant", constant_values=0)

            extended_A= torch.Tensor()
            extended_b= torch.Tensor()
            extended_G=torch.from_numpy(newG).float()
            extended_h=torch.from_numpy(newh).float()
            
            hess = -dual_hess.hess(xlamb, phis)
            regularization_term = 0.1 * torch.eye(hess.shape[-1])
            Q = hess + regularization_term

            jac = -dual_gradient(xlamb, phis)[:,:x_size + lamb_size]
            p = (jac.view(1, -1) - torch.matmul(xlamb, Q)).squeeze()
            
            qp_solver = qpthlocal.qp.QPFunction(verbose=True, solver=qpthlocal.qp.QPSolvers.GUROBI,
                                           zhats=xlamb)

            new_xlamb_opt = qp_solver(Q, p, extended_G, extended_h, extended_A, extended_b)
            new_x = new_xlamb_opt[:,:x_size]

            # print("Old xlamb")
            # print(xlamb)
            # print("New xlamb")
            # print(new_xlamb_opt)

            loss = -(labels.view(labels.shape[0], 1, labels.shape[1]).to("cpu") @ new_x.view(*new_x.shape, 1)).mean()
            print("Training loss: {}".format(loss))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_loss.append(loss.detach())

        print("Overall training loss: {}".format(np.mean(training_loss)))

        # ======================= testing ==========================
        testing_loss =[]
        for batch_idx, (features, labels) in enumerate(test_loader):
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            mean = model(features).view(nBatch, theta_size)
            variance = uncertainty_model(features).view(nBatch, theta_size)
            phis = torch.cat((-mean, variance), dim=1)
            x_lamb = torch.cat((x,lamb), dim=1)
            obj_value = dual_function(x_lamb, phis)

            def g(x):
                x_torch = torch.Tensor(x).view(1, x_size + lamb_size)
                value = -dual_function(x_torch, phis).detach().numpy()[0]
                # print(value)
                return value

            def g_jac(x):
                x_torch = torch.Tensor(x).view(1, x_size + lamb_size)
                gradient = -dual_gradient(x_torch, phis).detach().numpy()[0][:x_size + lamb_size]
                # gradient = -dual_function.get_jac_torch(x_torch, phi).detach().numpy()[0]
                return gradient

            def g_hess(x):
                x_torch = torch.Tensor(x).view(1, x_size + lamb_size)
                hess = dual_hess.hess(x_torch, phis)
                return -hess.detach().numpy()[0]

            def g_hessp(x, p):
                x_torch = torch.Tensor(x).view(1, x_size + lamb_size)
                p_torch = torch.Tensor(p)
                hessp = dual_hess.hessp(x_torch, phis, p_torch)
                return -hessp.detach().numpy()[0]

            start_time = time.time()
            res = scipy.optimize.minimize(fun=g, x0= np.random.rand((x_size + lamb_size)), method=method, jac=g_jac, hessp=g_hessp, bounds=[(-M, M)]*(x_size) + [(0.0, M)]*(lamb_size), constraints=constraints_slsqp, options={"maxiter": 10})

            xlamb = torch.Tensor(res.x).view(1, x_size + lamb_size)

            # newG = np.pad(G, ((0, lamb_size), (0, lamb_size)), "constant", constant_values=0)
            # newG[-lamb_size:, -lamb_size:] = -torch.eye(lamb_size)
            # newh = np.pad(h, (0, lamb_size), "constant", constant_values=0)

            # extended_A= torch.Tensor()
            # extended_b= torch.Tensor()
            # extended_G=torch.from_numpy(newG).float()
            # extended_h=torch.from_numpy(newh).float()
            # 
            # hess = -dual_hess.hess(xlamb, phis)
            # regularization_term = 0.1 * torch.eye(hess.shape[-1])
            # Q = hess + regularization_term

            # jac = -dual_gradient(xlamb, phis)[:,:x_size + lamb_size]
            # p = (jac.view(1, -1) - torch.matmul(xlamb, Q)).squeeze()
            # 
            # qp_solver = qpthlocal.qp.QPFunction(verbose=True, solver=qpthlocal.qp.QPSolvers.GUROBI,
            #                                zhats=xlamb)

            # new_xlamb_opt = qp_solver(Q, p, extended_G, extended_h, extended_A, extended_b)
            # new_x = new_xlamb_opt[:,:x_size]

            new_x = xlamb[:,:x_size]

            loss = -(labels.view(labels.shape[0], 1, labels.shape[1]).to("cpu") @ new_x.view(*new_x.shape, 1)).mean().detach()
            print("Testing loss: {}".format(loss))
            testing_loss.append(loss)

        print("Overall testing loss: {}".format(np.mean(testing_loss)))

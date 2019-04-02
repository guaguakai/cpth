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

from shortest_path_utils import Net, ShortestPathLoss, make_fc, constrained_attack, random_constraints

# from shortest_path_utils import load_data
# from shortest_path_utils import generate_graph
from shortest_path_utils import load_toy_data as load_data
from shortest_path_utils import generate_toy_graph as generate_graph

from toy_obj import Dual, DualFunction, DualGradient, DualHess
from linear import make_shortest_path_matrix

DTYPE = torch.float
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
                        help='number of epochs to train (default: 10)')
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

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # =============================================================================
    n_nodes = 4
    n_instances = 300
    n_features = 5
    graph, latency, source_list, dest_list = generate_graph(n_nodes=n_nodes, n_instances=n_instances)
    n_targets = graph.number_of_edges()

    # ========================= different constraints matrix ======================
    # constraint_matrix = 1.0 / self.x_size * np.random.normal(size=(self.m_size, self.x_size)) # random constraints
    # constraint_matrix = 1.0 / self.x_size * np.ones((1, self.x_size)) # budget constraint
    # constraint_matrix = np.concatenate((np.eye(self.x_size), -np.eye(self.x_size)), axis=0) # box constraints
    # constraint_matrix = np.concatenate((np.eye(self.x_size), -np.eye(self.x_size), 1.0 / self.x_size * np.random.normal(size=(self.m_size - 2 * self.x_size, self.x_size))), axis=0) # box constraints with random constraints

    constraint_matrix = np.concatenate((np.array([[1,0,0,0,0,0], [0,0,1,0,0,0], [0,0,0,1,0,0], [0,0,0,0,1,0]]), np.array([[0,1,0,0,0,1]])), axis=0) # box constraints with random constraints
    attacker_budget = np.array([0.1,0.0,0.0,0.1] + [0.3])
    # n_constraints = 10
    # constraint_matrix, attacker_budget = random_constraints(n_targets, n_constraints)

    # =============================== data loading ================================
    print("generating data...")
    train_loader, test_loader = load_data(args, kwargs, graph, latency, n_instances, n_features)

    edge_size = n_targets

    x_size     = edge_size
    theta_size = edge_size
    m_size = len(constraint_matrix)
    lamb_size  = m_size
    phi_size = edge_size + m_size
    M = 1e3
    tol = 1e-3
    method = "SLSQP"
    # method = "trust-constr"

    # =============================== models setup ================================
    model = Net(n_features, n_targets).to(device)
    uncertainty_model = Net(n_features, m_size).to(device)
    dual_function = DualFunction(model=model, x_size=x_size, theta_size=theta_size, m_size=m_size, edge_size=edge_size, phi_size=phi_size, constraint_matrix=constraint_matrix)
    dual_gradient = DualGradient(model=model, x_size=x_size, theta_size=theta_size, m_size=m_size, edge_size=edge_size, phi_size=phi_size, constraint_matrix=constraint_matrix)
    dual_hess     = DualHess(model=model, x_size=x_size, theta_size=theta_size, m_size=m_size, edge_size=edge_size, phi_size=phi_size, constraint_matrix=constraint_matrix)

    nBatch = args.batch_size
    print(nBatch)
    x = 1.0 * torch.ones((nBatch, x_size)) # TODO wrong dimension
    lamb = 0.1 * torch.ones((nBatch, lamb_size)) # TODO wrong dimension

    learning_rate = 1e-3
    num_epochs = args.epochs
    # num_epochs = args.epochs
    # optimizer = optim.Adam(list(model.parameters()) + list(uncertainty_model.parameters()), lr=learning_rate)

    # =========================== warm start -- two stage training ===================
    # print("training two stage...")
    # shortest_path_loss = ShortestPathLoss(n_nodes, graph, c, source_list, dest_list)
    # get_two_stage_loss = shortest_path_loss.get_two_stage_loss
    # loss_fn = get_two_stage_loss

    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.5)
    # for epoch in tqdm.trange(10):
    #     loss_list = []

    #     for batch_idx, (features, labels, indices) in enumerate(train_loader):
    #         features, labels = features.to(device), labels.to(device)
    #         batch_size = len(features)

    #         optimizer.zero_grad()
    #         loss = loss_fn(model, features, labels, indices, eval_mode=False)
    #         loss.backward()
    #         optimizer.step()

    #         if batch_idx % 10 == 0 and verbose:
    #             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
    #                 epoch, batch_idx * len(features), len(train_loader.dataset),
    #                 100. * batch_idx / len(train_loader), loss.item()))

    # ============================ main training part ===============================
    # optimizer = optim.SGD(list(model.parameters()), lr=learning_rate, momentum=0.5)
    # optimizer = optim.SGD(list(uncertainty_model.parameters()), lr=learning_rate, momentum=0.5)
    optimizer = optim.SGD(list(model.parameters()) + list(uncertainty_model.parameters()), lr=learning_rate, momentum=0.5)
    
    print("training...")
    for epoch in tqdm.trange(num_epochs):
        training_loss = []
        # ======================= training ==========================
        x0 = np.random.rand((x_size + lamb_size)) # initial point
        for batch_idx, (features, labels, indices) in enumerate(train_loader):
            features, labels = features.to(device), labels.to(device)
            # ------------------- shortest path matrix --------------
            source = source_list[indices]
            dest = dest_list[indices]
            A, b, G, h = make_shortest_path_matrix(graph, source, dest)

            newG = np.pad(G, ((0, lamb_size), (0, lamb_size)), "constant", constant_values=0)
            newG[-lamb_size:, -lamb_size:] = -torch.eye(lamb_size)
            newh = np.pad(h, (0, lamb_size), "constant", constant_values=0)
            newA = np.pad(A, ((0,0), (0, lamb_size)), "constant", constant_values=0)
            newb = np.pad(b, (0, lamb_size), "constant", constant_values=0)

            extended_A = torch.from_numpy(newA).float()
            extended_b = torch.from_numpy(newb).float()
            extended_G = torch.from_numpy(newG).float()
            extended_h = torch.from_numpy(newh).float()

            # -------------------- prediction ----------------------

            mean = model(features).view(nBatch, theta_size)

            # variance = torch.Tensor(attacker_budget).view(1, *attacker_budget.shape).cuda() # exact attacker budget 
            variance = uncertainty_model(features).view(nBatch, m_size) * 0.1 
            # print(variance.mean())
            phis = torch.cat((mean, variance), dim=1).cpu()
            x_lamb = torch.cat((x,lamb), dim=1)
            obj_value = dual_function(x_lamb, phis)
            #print ('labels ', labels)

            def ineq_fun(x):
                return G @ x[:x_size] - h

            def eq_fun(x):
                return A @ x[:x_size] - b

            constraints_slsqp = []
            constraints_slsqp.append({"type": "eq",   "fun": eq_fun,   "jac": autograd.jacobian(eq_fun)})
            
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
                hess = -dual_hess.hess(x_torch, phis)
                return hess.detach().numpy()[0]

            def g_hessp(x, p):
                x_torch = torch.Tensor(x).view(1, x_size + lamb_size)
                p_torch = torch.Tensor(p)
                hessp = -dual_hess.hessp(x_torch, phis, p_torch)
                return hessp.detach().numpy()[0]

            # start_time = time.time()
            # value = g(np.random.rand(x_size + lamb_size))
            # print("function evaluation time: {}".format(time.time() - start_time))

            # start_time = time.time()
            # value = g_jac(np.random.rand(x_size + lamb_size))
            # print("function gradient evaluation time: {}".format(time.time() - start_time))

            # start_time = time.time()
            # value = g_hess(np.random.rand(x_size + lamb_size))
            # print("Hessian evaluation time: {}".format(time.time() - start_time))


            res = scipy.optimize.minimize(fun=g, x0=x0, method=method, jac=g_jac, hessp=g_hessp, bounds=[(0.0, 1.0)]*(x_size) + [(0.0, M)]*(lamb_size), constraints=constraints_slsqp, options={"maxiter": 100})

            # x0 = res.x # update initial point
            x0 = np.random.rand((x_size + lamb_size)) # initial point

            xlamb = torch.Tensor(res.x).view(1, x_size + lamb_size)

            hess = -dual_hess.hess(xlamb, phis)
            regularization_term = 0.01 * torch.eye(hess.shape[-1])
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

            labels_modified = constrained_attack(new_x, labels, constraint_matrix, attacker_budget)

            loss = (labels_modified.view(labels_modified.shape[0], 1, labels.shape[1]).to("cpu") @ new_x.view(*new_x.shape, 1)).mean()
            if loss < 0:
                print("checking...")
                print(labels)
                print(new_x)
            # print("Training loss: {}".format(loss))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_loss.append(loss.detach())
            if batch_idx % 10 == 0 and verbose:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(features), len(train_loader),
                    100. * batch_idx / len(train_loader), np.mean(training_loss[-10:])))

        print("Overall training loss: {}".format(np.mean(training_loss)))

        # ======================= testing ==========================
        testing_loss =[]
        x0 = np.random.rand((x_size + lamb_size)) # initial point
        for batch_idx, (features, labels, indices) in enumerate(test_loader):
            features, labels = features.to(device), labels.to(device)
            # ------------------- shortest path matrix --------------
            source = source_list[indices]
            dest = dest_list[indices]
            A, b, G, h = make_shortest_path_matrix(graph, source, dest)

            newG = np.pad(G, ((0, lamb_size), (0, lamb_size)), "constant", constant_values=0)
            newG[-lamb_size:, -lamb_size:] = -torch.eye(lamb_size)
            newh = np.pad(h, (0, lamb_size), "constant", constant_values=0)
            newA = np.pad(A, ((0,0), (0, lamb_size)), "constant", constant_values=0)
            newb = np.pad(b, (0, lamb_size), "constant", constant_values=0)

            extended_A = torch.from_numpy(newA).float()
            extended_b = torch.from_numpy(newb).float()
            extended_G = torch.from_numpy(newG).float()
            extended_h = torch.from_numpy(newh).float()

            # -------------------- prediction ----------------------

            mean = model(features).view(nBatch, theta_size)
            # variance = torch.Tensor(attacker_budget).view(1, *attacker_budget.shape).cuda() # exact attacker budget 
            variance = uncertainty_model(features).view(nBatch, m_size).detach() * 0.1
            phis = torch.cat((mean, variance), dim=1).cpu().detach()
            x_lamb = torch.cat((x,lamb), dim=1)
            obj_value = dual_function(x_lamb, phis)

            def ineq_fun(x):
                return G @ x[:x_size] - h

            def eq_fun(x):
                return A @ x[:x_size] - b

            constraints_slsqp = []
            constraints_slsqp.append({"type": "eq",   "fun": eq_fun,   "jac": autograd.jacobian(eq_fun)})

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
                hess = -dual_hess.hess(x_torch, phis)
                return hess.detach().numpy()[0]

            def g_hessp(x, p):
                x_torch = torch.Tensor(x).view(1, x_size + lamb_size)
                p_torch = torch.Tensor(p)
                hessp = -dual_hess.hessp(x_torch, phis, p_torch)
                return hessp.detach().numpy()[0]

            start_time = time.time()
            res = scipy.optimize.minimize(fun=g, x0=x0, method=method, jac=g_jac, hessp=g_hessp, bounds=[(0.0, 1.0)]*(x_size) + [(0.0, M)]*(lamb_size), constraints=constraints_slsqp, options={"maxiter": 100})
            # x0 = res.x # update initial point
            x0 = np.random.rand((x_size + lamb_size)) # initial point

            xlamb = torch.Tensor(res.x).view(1, x_size + lamb_size)

            hess = -dual_hess.hess(xlamb, phis)
            regularization_term = 0.01 * torch.eye(hess.shape[-1])
            Q = hess + regularization_term

            jac = -dual_gradient(xlamb, phis)[:,:x_size + lamb_size]
            p = (jac.view(1, -1) - torch.matmul(xlamb, Q)).squeeze()
            
            qp_solver = qpthlocal.qp.QPFunction(verbose=True, solver=qpthlocal.qp.QPSolvers.GUROBI,
                                           zhats=xlamb)

            new_xlamb_opt = qp_solver(Q, p, extended_G, extended_h, extended_A, extended_b)
            new_x = new_xlamb_opt[:,:x_size]

            # new_x = xlamb[:,:x_size]
            labels_modified = constrained_attack(new_x, labels, constraint_matrix, attacker_budget)

            loss = (labels_modified.view(labels_modified.shape[0], 1, labels.shape[1]).to("cpu") @ new_x.view(*new_x.shape, 1)).mean().detach()
            # print("source: {}, destination: {}, loss: {}, decision: {}, latency: {}".format(source, dest, loss, new_x, labels_modified))
            if loss < 0:
                print("checking...")
                print(labels)
                print(new_x)
            # print("Testing loss: {}".format(loss))
            testing_loss.append(loss)

        print("Overall testing loss: {}".format(np.mean(testing_loss)))

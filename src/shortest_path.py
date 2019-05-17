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

from shortest_path_utils import load_data
from shortest_path_utils import generate_graph
# from shortest_path_utils import load_toy_data as load_data
# from shortest_path_utils import generate_toy_graph as generate_graph

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
    parser.add_argument('--epochs', type=int, default=2, metavar='N',
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

    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

    # =============================================================================
    n_nodes = 10
    n_instances = 500
    n_features = 5
    graph, latency, source_list, dest_list = generate_graph(n_nodes=n_nodes, n_instances=n_instances)
    n_targets = graph.number_of_edges()

    # ========================= different constraints matrix ======================
    # constraint_matrix = 1.0 / self.x_size * np.random.normal(size=(self.m_size, self.x_size)) # random constraints
    # constraint_matrix = 1.0 / self.x_size * np.ones((1, self.x_size)) # budget constraint
    # constraint_matrix = np.concatenate((np.eye(self.x_size), -np.eye(self.x_size)), axis=0) # box constraints
    # constraint_matrix = np.concatenate((np.eye(self.x_size), -np.eye(self.x_size), 1.0 / self.x_size * np.random.normal(size=(self.m_size - 2 * self.x_size, self.x_size))), axis=0) # box constraints with random constraints

    # constraint_matrix = np.concatenate((np.array([[1,0,0,0,0,0], [0,0,1,0,0,0], [0,0,0,1,0,0], [0,0,0,0,1,0]]), np.array([[0,1,0,0,0,1]])), axis=0) # box constraints with random constraints
    # attacker_budgets = np.array([0.1,0.0,0.0,0.1] + [0.3])
    n_constraints = 10
    # constraint_matrix, attacker_budgets = random_constraints(n_targets, n_constraints)

    # =============================== data loading ================================
    print("generating data...")
    max_budget = 1.0
    train_loader, test_loader, constraint_matrix = load_data(args, kwargs, graph, latency, n_instances, n_constraints, n_features=n_features, max_budget=max_budget)

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
    learning_rate = 1e-4
    num_epochs = args.epochs

    f_ts_loss = open("exp/ts/loss_0516.csv", "w")
    f_ts_obj = open("exp/ts/obj_0516.csv", "w")
    f_df_loss = open("exp/df/loss_0516.csv", "w")
    f_df_obj = open("exp/df/obj_0516.csv", "w")


    # ==============================================================================
    #              #  #       #       #####              #####    ####
    #              # #       # #        #                  #     #
    #              ##       #   #       #                  #      ####
    #              # #     #######      #                  #          #
    #              #  #   #       #   #####                #      ####
    # ==============================================================================

    # ============================ two-stage models setup ================================
    model_ts = Net(n_features, n_targets).to(device)
    uncertainty_model_ts = Net(n_features, m_size).to(device)

    dual_function = DualFunction(x_size=x_size, theta_size=theta_size, m_size=m_size, edge_size=edge_size, phi_size=phi_size, constraint_matrix=constraint_matrix)
    dual_gradient = DualGradient(x_size=x_size, theta_size=theta_size, m_size=m_size, edge_size=edge_size, phi_size=phi_size, constraint_matrix=constraint_matrix)
    dual_hess     = DualHess(x_size=x_size, theta_size=theta_size, m_size=m_size, edge_size=edge_size, phi_size=phi_size, constraint_matrix=constraint_matrix)

    nBatch = args.batch_size # for computing
    batch_size = 10 # for loss back propagation

    print(nBatch)
    x = 1.0 * torch.ones((nBatch, x_size)) # TODO wrong dimension
    lamb = 0.1 * torch.ones((nBatch, lamb_size)) # TODO wrong dimension

    # ============================= two stage training ================================
    print("training two stage...")
    optimizer = optim.Adam(list(model_ts.parameters()) + list(uncertainty_model_ts.parameters()), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()

    for epoch in tqdm.trange(num_epochs):
        for mode in ["training", "testing"]:
            print("{}...".format(mode))
            data_loader = train_loader if mode == "training" else test_loader
            if mode == "training":
                model_ts.train()
                uncertainty_model_ts.train()
            else:
                model_ts.eval()
                uncertainty_model_ts.eval()

            loss_list = []
            obj_list = []
            batch_loss = 0

            x0 = np.random.rand((x_size + lamb_size)) # initial point
            for batch_idx, (features, labels, attacker_budgets, indices) in enumerate(data_loader):
                features, labels, attacker_budgets = features.to(device), labels.to(device), attacker_budgets.to(device)
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

                mean = model_ts(features).view(nBatch, theta_size)
                # variance = torch.Tensor(attacker_budgets).view(1, *attacker_budgets.shape).cuda() # exact attacker budget 
                variance = uncertainty_model_ts(features).view(nBatch, m_size)
                phis = torch.cat((mean, variance), dim=1).cpu().detach()
                loss = loss_fn(mean, labels) + loss_fn(variance, attacker_budgets)
                batch_loss += loss

                if mode == "training" and batch_idx % batch_size == batch_size-1:
                    optimizer.zero_grad()
                    batch_loss.backward()
                    optimizer.step()
                    batch_loss = 0

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
                labels_modified = constrained_attack(new_x, labels, constraint_matrix, attacker_budgets)
                obj_value = (labels_modified.view(labels_modified.shape[0], 1, labels.shape[1]).to("cpu") @ new_x.view(*new_x.shape, 1)).mean()

                loss_list.append(loss.item())
                obj_list.append(obj_value.item())

                if (batch_idx+1) % 10 == 0 and verbose:
                    print('{} Epoch: {} [{}/{} ({:.0f}%)]\t Average Loss: {:.6f}, Average obj value: {}'.format(
                        mode, epoch, (batch_idx+1) * len(features), len(data_loader),
                        100. * batch_idx / len(data_loader), np.mean(loss_list[-10:]), np.mean(obj_list[-10:])))

            print("Overall {} loss: {}, obj value: {}".format(mode, np.mean(loss_list), np.mean(obj_list)))
            f_ts_loss.write("Epoch, {}, mode, {}, loss, {}, loss std, {} \n".format(epoch, mode, np.mean(loss_list), np.std(loss_list)))
            f_ts_obj.write("Epoch, {}, mode, {}, obj values, {}, obj std, {} \n".format(epoch, mode, np.mean(obj_list), np.std(obj_list)))


    # ==============================================================================
    #              #  #       #       #####              ####    #####
    #              # #       # #        #                #   #   #
    #              ##       #   #       #                #   #   #####
    #              # #     #######      #                #   #   #  
    #              #  #   #       #   #####              ####    #
    # ======================== decision-focused models setup =============================
    model = Net(n_features, n_targets).to(device)
    uncertainty_model = Net(n_features, m_size).to(device)
    dual_function = DualFunction(x_size=x_size, theta_size=theta_size, m_size=m_size, edge_size=edge_size, phi_size=phi_size, constraint_matrix=constraint_matrix)
    dual_gradient = DualGradient(x_size=x_size, theta_size=theta_size, m_size=m_size, edge_size=edge_size, phi_size=phi_size, constraint_matrix=constraint_matrix)
    dual_hess     = DualHess(x_size=x_size, theta_size=theta_size, m_size=m_size, edge_size=edge_size, phi_size=phi_size, constraint_matrix=constraint_matrix)

    x = 1.0 * torch.ones((nBatch, x_size)) # TODO wrong dimension
    lamb = 0.1 * torch.ones((nBatch, lamb_size)) # TODO wrong dimension

    # ============================ main training part ===============================
    # optimizer = optim.SGD(list(model.parameters()), lr=learning_rate, momentum=0.5)
    # optimizer = optim.SGD(list(uncertainty_model.parameters()), lr=learning_rate, momentum=0.5)
    print("end-to-end training...")
    optimizer = optim.SGD(list(model.parameters()) + list(uncertainty_model.parameters()), lr=learning_rate, momentum=0.5)
    
    for epoch in tqdm.trange(num_epochs):
        # ======================= training and testing ==========================
        for mode in ["training", "testing"]:
            print("{}...".format(mode))
            data_loader = train_loader if mode == "training" else test_loader
            if mode == "training":
                model_ts.train()
                uncertainty_model_ts.train()
            else:
                model_ts.eval()
                uncertainty_model_ts.eval()

            loss_list = []
            obj_list = []
            batch_loss = 0

            x0 = np.random.rand((x_size + lamb_size)) # initial point
            for batch_idx, (features, labels, attacker_budgets, indices) in enumerate(data_loader):
                features, labels, attacker_budgets = features.to(device), labels.to(device), attacker_budgets.to(device)
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
                # variance = torch.Tensor(attacker_budgets).view(1, *attacker_budgets.shape).cuda() # exact attacker budget 
                variance = uncertainty_model(features).view(nBatch, m_size)
                loss = loss_fn(mean, labels) + loss_fn(variance, attacker_budgets)
                loss_list.append(loss.item())

                phis = torch.cat((mean, variance), dim=1).cpu()

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

                labels_modified = constrained_attack(new_x, labels, constraint_matrix, attacker_budgets)

                obj_value = (labels_modified.view(labels_modified.shape[0], 1, labels.shape[1]).to("cpu") @ new_x.view(*new_x.shape, 1)).mean()
                obj_list.append(obj_value.item())
                batch_loss += obj_value
                if obj_value < 0:
                    print("checking...")
                    print(labels)
                    print(new_x)
                # print("Training loss: {}".format(loss))

                if mode == "training" and (batch_idx % batch_size) == batch_size-1:
                    optimizer.zero_grad()
                    batch_loss.backward()
                    optimizer.step()
                    batch_loss = 0

                if (batch_idx+1) % 10 == 0 and verbose:
                    print('{} Epoch: {} [{}/{} ({:.0f}%)]\tAverage loss: {:.6f}, Average obj value: {}'.format(
                        mode, epoch, (batch_idx+1) * len(features), len(data_loader),
                        100. * batch_idx / len(data_loader), np.mean(loss_list[-10:]), np.mean(obj_list[-10:])))

            print("Overall {} loss: {}, obj value: {}".format(mode, np.mean(loss_list), np.mean(obj_list)))
            f_df_loss.write("Epoch, {}, mode, {}, loss, {}, loss std, {} \n".format(epoch, mode, np.mean(loss_list), np.std(loss_list)))
            f_df_obj.write("Epoch, {}, mode, {}, obj values, {}, obj std, {} \n".format(epoch, mode, np.mean(obj_list), np.std(obj_list)))

    f_ts_loss.close()
    f_ts_obj.close()
    f_df_loss.close()
    f_df_obj.close()

        # # ======================= testing ==========================
        # testing_loss =[]
        # x0 = np.random.rand((x_size + lamb_size)) # initial point
        # for batch_idx, (features, labels, attacker_budgets, indices) in enumerate(test_loader):
        #     features, labels = features.to(device), labels.to(device)
        #     # ------------------- shortest path matrix --------------
        #     source = source_list[indices]
        #     dest = dest_list[indices]
        #     A, b, G, h = make_shortest_path_matrix(graph, source, dest)

        #     newG = np.pad(G, ((0, lamb_size), (0, lamb_size)), "constant", constant_values=0)
        #     newG[-lamb_size:, -lamb_size:] = -torch.eye(lamb_size)
        #     newh = np.pad(h, (0, lamb_size), "constant", constant_values=0)
        #     newA = np.pad(A, ((0,0), (0, lamb_size)), "constant", constant_values=0)
        #     newb = np.pad(b, (0, lamb_size), "constant", constant_values=0)

        #     extended_A = torch.from_numpy(newA).float()
        #     extended_b = torch.from_numpy(newb).float()
        #     extended_G = torch.from_numpy(newG).float()
        #     extended_h = torch.from_numpy(newh).float()

        #     # -------------------- prediction ----------------------
        #     mean = model(features).view(nBatch, theta_size)
        #     # variance = torch.Tensor(attacker_budgets).view(1, *attacker_budgets.shape).cuda() # exact attacker budget 
        #     variance = uncertainty_model(features).view(nBatch, m_size).detach()
        #     phis = torch.cat((mean, variance), dim=1).cpu().detach()

        #     def ineq_fun(x):
        #         return G @ x[:x_size] - h

        #     def eq_fun(x):
        #         return A @ x[:x_size] - b

        #     constraints_slsqp = []
        #     constraints_slsqp.append({"type": "eq",   "fun": eq_fun,   "jac": autograd.jacobian(eq_fun)})

        #     def g(x):
        #         x_torch = torch.Tensor(x).view(1, x_size + lamb_size)
        #         value = -dual_function(x_torch, phis).detach().numpy()[0]
        #         # print(value)
        #         return value

        #     def g_jac(x):
        #         x_torch = torch.Tensor(x).view(1, x_size + lamb_size)
        #         gradient = -dual_gradient(x_torch, phis).detach().numpy()[0][:x_size + lamb_size]
        #         # gradient = -dual_function.get_jac_torch(x_torch, phi).detach().numpy()[0]
        #         return gradient

        #     def g_hess(x):
        #         x_torch = torch.Tensor(x).view(1, x_size + lamb_size)
        #         hess = -dual_hess.hess(x_torch, phis)
        #         return hess.detach().numpy()[0]

        #     def g_hessp(x, p):
        #         x_torch = torch.Tensor(x).view(1, x_size + lamb_size)
        #         p_torch = torch.Tensor(p)
        #         hessp = -dual_hess.hessp(x_torch, phis, p_torch)
        #         return hessp.detach().numpy()[0]

        #     start_time = time.time()
        #     res = scipy.optimize.minimize(fun=g, x0=x0, method=method, jac=g_jac, hessp=g_hessp, bounds=[(0.0, 1.0)]*(x_size) + [(0.0, M)]*(lamb_size), constraints=constraints_slsqp, options={"maxiter": 100})
        #     # x0 = res.x # update initial point
        #     x0 = np.random.rand((x_size + lamb_size)) # initial point

        #     xlamb = torch.Tensor(res.x).view(1, x_size + lamb_size)

        #     hess = -dual_hess.hess(xlamb, phis)
        #     regularization_term = 0.01 * torch.eye(hess.shape[-1])
        #     Q = hess + regularization_term

        #     jac = -dual_gradient(xlamb, phis)[:,:x_size + lamb_size]
        #     p = (jac.view(1, -1) - torch.matmul(xlamb, Q)).squeeze()
        #     
        #     qp_solver = qpthlocal.qp.QPFunction(verbose=True, solver=qpthlocal.qp.QPSolvers.GUROBI,
        #                                    zhats=xlamb)

        #     new_xlamb_opt = qp_solver(Q, p, extended_G, extended_h, extended_A, extended_b)
        #     new_x = new_xlamb_opt[:,:x_size]

        #     # new_x = xlamb[:,:x_size]
        #     labels_modified = constrained_attack(new_x, labels, constraint_matrix, attacker_budgets)

        #     loss = (labels_modified.view(labels_modified.shape[0], 1, labels.shape[1]).to("cpu") @ new_x.view(*new_x.shape, 1)).mean().detach()
        #     # print("source: {}, destination: {}, loss: {}, decision: {}, latency: {}".format(source, dest, loss, new_x, labels_modified))
        #     if loss < 0:
        #         print("checking...")
        #         print(labels)
        #         print(new_x)
        #     # print("Testing loss: {}".format(loss))
        #     testing_loss.append(loss)

        # print("Overall testing loss: {}".format(np.mean(testing_loss)))

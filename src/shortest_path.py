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
verbose_debug = False
D_CONST = 0
D_ABNORMAL = 1.0


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Matching')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                        help='input batch size for testing (default: 1)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--nonrobust', dest='robust', action='store_false', default=True,
                        help='disable robust learning')

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)

    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

    # =============================================================================
    n_nodes = 20
    n_instances = 300
    n_features = 10
    graph, latency, source_list, dest_list = generate_graph(n_nodes=n_nodes, n_instances=n_instances)
    n_targets = graph.number_of_edges()
    n_constraints = 5

    # =============================== data loading ================================
    print("generating data...")
    max_budget  = 0.5
    max_latency = 0.5
    train_loader, test_loader, constraint_matrix = load_data(args, kwargs, graph, latency, n_instances, n_constraints, n_features=n_features, max_budget=max_budget)

    edge_size = n_targets

    x_size     = edge_size
    theta_size = edge_size
    m_size = len(constraint_matrix)
    lamb_size  = m_size
    phi_size = edge_size + m_size
    M = 10
    tol = 1e-3
    method = "SLSQP"
    # method = "trust-constr"
    learning_rate = args.lr
    num_epochs = args.epochs

    nBatch = args.batch_size # for computing
    batch_size = 10 # for loss back propagation
    loss_fn = torch.nn.MSELoss()

    # ======================= setting for robust learning ==========================
    #                   ####     ###    ###    #   #    ###   #####
    #                   #   #   #   #   #  #   #   #   #        #
    #                   ####    #   #   ###    #   #    ###     #
    #                   #   #   #   #   #  #   #   #       #    #
    #                   #   #    ###    ###     ###     ###     #
    # ==============================================================================
    robust_option = args.robust
    print("Enable robust optimization: {}".format(robust_option))

    # ================================= filename ===================================
    folder_path = "exp/robust/" if robust_option else "exp/nonrobust/"
    filename = "0523_node{}_const{}_feat{}".format(n_nodes, n_constraints, n_features)
    f_ts_loss = open(folder_path + "ts/loss_{}.csv".format(filename), "w")
    f_ts_obj  = open(folder_path + "ts/obj_{}.csv".format(filename), "w")
    f_df_loss = open(folder_path + "df/loss_{}.csv".format(filename), "w")
    f_df_obj  = open(folder_path + "df/obj_{}.csv".format(filename), "w")


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

    # ==============================================================================
    #              #  #       #       #####              ####    #####
    #              # #       # #        #                #   #   #
    #              ##       #   #       #                #   #   #####
    #              # #     #######      #                #   #   #  
    #              #  #   #       #   #####              ####    #
    # ======================== decision-focused models setup =============================
    model_df = Net(n_features, n_targets).to(device)
    uncertainty_model_df = Net(n_features, m_size).to(device)

    # ============================= dual function setip ==================================
    dual_function = DualFunction(x_size=x_size, theta_size=theta_size, m_size=m_size, edge_size=edge_size, phi_size=phi_size, constraint_matrix=constraint_matrix)
    dual_gradient = DualGradient(x_size=x_size, theta_size=theta_size, m_size=m_size, edge_size=edge_size, phi_size=phi_size, constraint_matrix=constraint_matrix)
    dual_hess     = DualHess(x_size=x_size, theta_size=theta_size, m_size=m_size, edge_size=edge_size, phi_size=phi_size, constraint_matrix=constraint_matrix)

    x = 1.0 * torch.ones((nBatch, x_size)) # TODO wrong dimension
    lamb = 0.1 * torch.ones((nBatch, lamb_size)) # TODO wrong dimension

    # """
    # ============================= two stage training ================================
    for training_option in ["two-stage", "decision-focused"]:
    # for training_option in ["decision-focused"]:
        print("Training {}...".format(training_option))
        f_loss = f_ts_loss if training_option == "two-stage" else f_df_loss
        f_obj  = f_ts_obj  if training_option == "two-stage" else f_df_obj
        model = model_ts if training_option == "two-stage" else model_df
        uncertainty_model = uncertainty_model_ts if training_option == "two-stage" else uncertainty_model_df
        optimizer = optim.Adam(list(model.parameters()) + list(uncertainty_model.parameters()), lr=learning_rate)

        for epoch in tqdm.trange(-2, num_epochs):
            for mode in ["training", "testing"]:
                print("{}...".format(mode))
                data_loader = train_loader if mode == "training" else test_loader
                if mode == "training" and epoch >= 0:
                    model.train()
                    uncertainty_model.train()
                else:
                    model.eval()
                    uncertainty_model.eval()

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

                    newG = np.pad(G, ((0, lamb_size * 2), (0, lamb_size)), "constant", constant_values=0)
                    newG[-2*lamb_size:-lamb_size, -lamb_size:] = -torch.eye(lamb_size)
                    newG[-lamb_size:,-lamb_size:] = torch.eye(lamb_size)
                    newh = np.pad(h, (0, lamb_size * 2), "constant", constant_values=D_CONST)
                    newh[-lamb_size:] = M
                    newA = np.pad(A, ((0,0), (0, lamb_size)), "constant", constant_values=0)
                    newb = b # np.pad(b, (0, lamb_size), "constant", constant_values=0)

                    extended_A = torch.from_numpy(newA).float()
                    extended_b = torch.from_numpy(newb).float()
                    extended_G = torch.from_numpy(newG).float()
                    extended_h = torch.from_numpy(newh).float()

                    # -------------------- prediction ----------------------

                    mean = model(features).view(nBatch, theta_size) * max_latency
                    if epoch == -2: # check the optimal performance
                        mean = labels
                        variance = attacker_budgets
                    elif robust_option:
                        variance = torch.zeros(nBatch, m_size).to(device)
                        variance[:,:-n_targets] = uncertainty_model(features).view(nBatch, m_size)[:,:-n_targets] * max_budget
                        variance[:,-n_targets] = 0
                    else:
                        variance = torch.zeros(nBatch, m_size).to(device)

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

                    res = scipy.optimize.minimize(fun=g, x0=x0, method=method, jac=g_jac, hessp=g_hessp, bounds=[(0.0, 1.0)]*(x_size) + [(0.0, M)]*(lamb_size), constraints=constraints_slsqp, options={"maxiter": 100})

                    # x0 = res.x # update initial point
                    x0 = np.random.rand((x_size + lamb_size)) # initial point

                    xlamb = torch.Tensor(res.x).view(1, x_size + lamb_size)

                    hess = -dual_hess.hess(xlamb, phis)
                    regularization_term = 0.01 * torch.eye(hess.shape[-1])
                    Q = hess + regularization_term

                    jac = -dual_gradient(xlamb, phis)[:,:x_size + lamb_size]
                    p = (jac.view(1, -1) - torch.matmul(xlamb, Q)).squeeze()
                    
                    qp_solver = qpth.qp.QPFunction()
                    # qp_solver = qpthlocal.qp.QPFunction(verbose=True, solver=qpthlocal.qp.QPSolvers.GUROBI, zhats=xlamb)

                    new_xlamb_opt = qp_solver(Q, p, extended_G, extended_h, extended_A, extended_b)
                    # new_x = xlamb[:,:x_size]
                    new_x = new_xlamb_opt[:,:x_size]

                    labels_modified = constrained_attack(new_x, labels, constraint_matrix, attacker_budgets)
                    obj_value = (labels_modified.view(labels_modified.shape[0], 1, labels.shape[1]).to("cpu") @ new_x.view(*new_x.shape, 1)).mean()

                    if torch.norm(xlamb[:,:x_size] - new_x) > D_ABNORMAL and verbose_debug:
                        print("ABNORMAL NORM: {}".format(torch.norm(xlamb[:,:x_size] - new_x)))
                        print(res)
                        print(source, dest)
                        print("mean: {}, var: {}".format(mean, variance))
                        print("label: {}, attacker budget: {}".format(labels, attacker_budgets))
                        print("Old xlamb")
                        print(xlamb)
                        print("New xlamb")
                        print(new_xlamb_opt)
                        print(labels_modified)

                    if torch.any(obj_value < - D_CONST):
                        print("checking...")
                        print(labels)
                        print(new_x)

                    if robust_option:
                        loss = loss_fn(mean, labels) + loss_fn(variance, attacker_budgets)
                    else:
                        loss = loss_fn(mean, labels_modified) 

                    batch_loss += loss if training_option == "two-stage" else obj_value # loss calculation

                    if mode == "training" and epoch >= 0 and batch_idx % batch_size == batch_size-1:
                        optimizer.zero_grad()
                        batch_loss.backward()
                        optimizer.step()
                        batch_loss = 0

                    loss_list.append(loss.item())
                    obj_list.append(obj_value.item())

                    if (batch_idx+1) % 10 == 0 and verbose:
                        print('{} Epoch: {} [{}/{} ({:.0f}%)]\t Average Loss: {:.6f}, Average obj value: {}'.format(
                            mode, epoch, (batch_idx+1) * len(features), len(data_loader),
                            100. * batch_idx / len(data_loader), np.mean(loss_list[-10:]), np.mean(obj_list[-10:])))

                print("Overall {} loss: {}, obj value: {}".format(mode, np.mean(loss_list), np.mean(obj_list)))
                f_loss.write("Epoch, {}, mode, {}, loss, {}, loss std, {} \n".format(epoch, mode, np.mean(loss_list), np.std(loss_list)))
                f_obj.write("Epoch, {}, mode, {}, obj values, {}, obj std, {} \n".format(epoch, mode, np.mean(obj_list), np.std(obj_list)))


    """
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
    print("end-to-end training...")
    optimizer = optim.Adam(list(model.parameters()) + list(uncertainty_model.parameters()), lr=learning_rate)
    
    for epoch in tqdm.trange(-2, num_epochs):
        # ======================= training and testing ==========================
        for mode in ["training", "testing"]:
            print("{}...".format(mode))
            data_loader = train_loader if mode == "training" else test_loader
            if mode == "training" and epoch >= 0:
                model.train()
                uncertainty_model.train()
            else:
                model.eval()
                uncertainty_model.eval()

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
                newh = np.pad(h, (0, lamb_size), "constant", constant_values=D_CONST)
                newA = np.pad(A, ((0,0), (0, lamb_size)), "constant", constant_values=0)
                newb = b # np.pad(b, (0, lamb_size), "constant", constant_values=0)

                extended_A = torch.from_numpy(newA).float()
                extended_b = torch.from_numpy(newb).float()
                extended_G = torch.from_numpy(newG).float()
                extended_h = torch.from_numpy(newh).float()

                # -------------------- prediction ----------------------

                mean = model(features).view(nBatch, theta_size) * max_latency
                # mean = labels
                if epoch == -2: # check the optimal performance
                    mean = labels
                    variance = attacker_budgets
                elif robust_option:
                    variance = torch.zeros(nBatch, m_size).to(device)
                    variance[:,:-n_targets] = uncertainty_model(features).view(nBatch, m_size)[:,:-n_targets] * max_budget
                    variance[:,-n_targets] = 0
                    # mean = labels # FOR TESTING ONLY
                    # variance = attacker_budgets # FOR TESTING ONLY
                else:
                    variance = torch.zeros(nBatch, m_size).to(device)

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

                res = scipy.optimize.minimize(fun=g, x0=x0, method=method, jac=g_jac, hessp=g_hessp, bounds=[(0.0, 1.0)]*(x_size) + [(0.0, M)]*(lamb_size), constraints=constraints_slsqp, options={"maxiter": 100})

                # x0 = res.x # update initial point
                x0 = np.random.rand((x_size + lamb_size)) # initial point

                xlamb = torch.Tensor(res.x).view(1, x_size + lamb_size)

                hess = -dual_hess.hess(xlamb, phis)
                regularization_term = 0.05 * torch.eye(hess.shape[-1])
                Q = hess + regularization_term

                jac = -dual_gradient(xlamb, phis)[:,:x_size + lamb_size]
                p = (jac.view(1, -1) - torch.matmul(xlamb, Q)).squeeze()
                
                qp_solver = qpth.qp.QPFunction()
                # qp_solver = qpthlocal.qp.QPFunction(verbose=True, solver=qpthlocal.qp.QPSolvers.GUROBI, zhats=xlamb)

                new_xlamb_opt = qp_solver(Q, p, extended_G, extended_h, extended_A, extended_b)
                new_x = new_xlamb_opt[:,:x_size] # the new x is not perfectly aligned with the old x. It should be. #TODO

                labels_modified = constrained_attack(new_x, labels, constraint_matrix, attacker_budgets)

                if robust_option:
                    loss = loss_fn(mean, labels) + loss_fn(variance, attacker_budgets)
                else:
                    loss = loss_fn(mean, labels_modified) 

                obj_value = (labels_modified.view(labels_modified.shape[0], 1, labels.shape[1]).to("cpu") @ new_x.view(*new_x.shape, 1)).mean().to(device)

                if torch.norm(xlamb[:,:x_size] - new_x) > D_ABNORMAL and verbose_debug:
                    print("ABNORMAL NORM: {}".format(torch.norm(xlamb[:,:x_size] - new_x)))
                    print(source, dest)
                    print("mean: {}, var: {}".format(mean, variance))
                    print("label: {}, attacker budget: {}".format(labels, attacker_budgets))
                    print("Old xlamb")
                    print(xlamb)
                    print("New xlamb")
                    print(new_xlamb_opt)
                    print(labels_modified)

                if torch.any(obj_value < - D_CONST):
                    print("checking...")
                    print(labels)
                    print(new_x)

                loss_list.append(loss.item())
                obj_list.append(obj_value.item())

                batch_loss += obj_value

                if mode == "training" and epoch >= 0 and (batch_idx % batch_size) == batch_size-1:
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

    #"""
    f_ts_loss.close()
    f_ts_obj.close()
    f_df_loss.close()
    f_df_obj.close()


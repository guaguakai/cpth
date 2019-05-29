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
import itertools

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

from toy_obj import Dual, DualFunction, DualGradient, DualHess
from linear import make_shortest_path_matrix

DTYPE = torch.float
visualization = False
verbose = True
verbose_debug = True
D_CONST = 0
D_ABNORMAL = 1.0
D_EPSILON = 1.0


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Matching')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                        help='input batch size for testing (default: 1)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001)')
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

    # ================================= toy example ===============================
    toy_option = False
    if toy_option:
        from shortest_path_utils import load_toy_data as load_data
        from shortest_path_utils import generate_toy_graph as generate_graph
        n_nodes = 4
        n_instances = 300
        n_features = 5
        p = 0.3 # edge prob
        max_budget  = 0.5
        max_latency = 0.5
        n_constraints = 5
    else:
        from shortest_path_utils import load_data
        from shortest_path_utils import generate_graph_erdos as generate_graph
        # from shortest_path_utils import generate_graph_geometric as generate_graph
        n_nodes = 10
        n_instances = 300
        n_features = 128
        p = 0.3 # edge prob
        max_budget  = 5.0
        max_latency = 5.0
        intermediate_size = 512
        n_constraints = 10

    graph, latency, source_list, dest_list = generate_graph(n_nodes=n_nodes, p=p, n_instances=n_instances)
    n_targets = graph.number_of_edges()

    # =============================== data loading ================================
    print("generating data...")
    train_loader, test_loader, constraint_matrix = load_data(args, kwargs, graph, latency, n_instances, n_constraints, n_features=n_features, max_budget=max_budget)

    edge_size = n_targets

    x_size     = edge_size
    theta_size = edge_size
    m_size = len(constraint_matrix)
    lamb_size  = m_size
    phi_size = edge_size + m_size
    M = 1e4
    tol = 1e-3
    method = "SLSQP"
    # method = "trust-constr"
    learning_rate = float(args.lr)
    num_epochs = args.epochs

    nBatch = args.batch_size # for computing
    batch_size = 1 # for loss back propagation
    loss_fn = torch.nn.MSELoss()

    # ======================= setting for robust learning ==========================
    #                   ####     ###    ###    #   #    ###   #####
    #                   #   #   #   #   #  #   #   #   #        #
    #                   ####    #   #   ###    #   #    ###     #
    #                   #   #   #   #   #  #   #   #       #    #
    #                   #   #    ###    ###     ###     ###     #
    # ==============================================================================
    blackbox_option = False
    if not blackbox_option: # precompute
        relaxation = 0.01
        regularization = 0.01

        P_inv = (1/relaxation) * np.eye(theta_size)
        Q = np.zeros((1, x_size + lamb_size, x_size + lamb_size))
        Q[0,:x_size, :x_size] += P_inv
        Q[0,:x_size, x_size:] += - np.transpose(constraint_matrix @ P_inv)
        Q[0,x_size:, :x_size] += - constraint_matrix @ P_inv
        Q[0,x_size:, x_size:] += constraint_matrix @ P_inv @ np.transpose(constraint_matrix)
        Q = torch.Tensor(Q).to("cpu")
        Q = Q + regularization * torch.eye(x_size + lamb_size)



    # ================================= filename ===================================
    # folder_path = "exp/robust/" if robust_option else "exp/nonrobust/"
    filename = "0528_1900_server_node{}_const{}_feat{}".format(n_nodes, n_constraints, n_features)
    # f_ts_loss = open(folder_path + "ts/loss_{}.csv".format(filename), "w")
    # f_ts_obj  = open(folder_path + "ts/obj_{}.csv".format(filename), "w")
    # f_df_loss = open(folder_path + "df/loss_{}.csv".format(filename), "w")
    # f_df_obj  = open(folder_path + "df/obj_{}.csv".format(filename), "w")

    f_summary_obj = open("exp/summary/obj_{}.csv".format(filename), "a")
    f_summary_loss = open("exp/summary/loss_{}.csv".format(filename), "a")


    # ==============================================================================
    #              #  #       #       #####              #####    ####
    #              # #       # #        #                  #     #
    #              ##       #   #       #                  #      ####
    #              # #     #######      #                  #          #
    #              #  #   #       #   #####                #      ####
    # ==============================================================================

    # ==============================================================================
    #              #  #       #       #####              ####    #####
    #              # #       # #        #                #   #   #
    #              ##       #   #       #                #   #   #####
    #              # #     #######      #                #   #   #  
    #              #  #   #       #   #####              ####    #
    # ============================ two-stage models setup ================================
    # ======================== decision-focused models setup =============================

    # ============================= dual function setip ==================================
    dual_function = DualFunction(x_size=x_size, theta_size=theta_size, m_size=m_size, edge_size=edge_size, phi_size=phi_size, constraint_matrix=constraint_matrix)
    dual_gradient = DualGradient(x_size=x_size, theta_size=theta_size, m_size=m_size, edge_size=edge_size, phi_size=phi_size, constraint_matrix=constraint_matrix)
    dual_hess     = DualHess(x_size=x_size, theta_size=theta_size, m_size=m_size, edge_size=edge_size, phi_size=phi_size, constraint_matrix=constraint_matrix)

    # """
    training_loss = np.zeros((4, num_epochs + 2)) # robust, nonrobust; ts, df
    testing_loss  = np.zeros((4, num_epochs + 2))
    training_obj  = np.zeros((4, num_epochs + 2))
    testing_obj   = np.zeros((4, num_epochs + 2))

    # ============================ model initialization ===============================
    model_initial = Net(n_features, n_targets).to(device)
    uncertainty_model_initial = Net(n_features, m_size).to(device)

    # ============================= two stage training ================================
    # for idx, (robust_option, training_option) in enumerate(itertools.product([True], ["two-stage", "decision-focused"])):
    for idx, (robust_option, training_option) in enumerate(itertools.product([False, True], ["two-stage", "decision-focused"])):
        print("Training {} {}...".format("robust" if robust_option else "non-robust", training_option))

        model = copy.deepcopy(model_initial)
        uncertainty_model = copy.deepcopy(uncertainty_model_initial)

        optimizer = optim.Adam(list(model.parameters()) + list(uncertainty_model.parameters()), lr=learning_rate)

        for epoch in tqdm.trange(-2, num_epochs): # start from -1 to test original objective value, -2 to test the optimal
            for mode in ["training", "testing"]:
                loss_record = training_loss if mode == "training" else testing_loss
                obj_record  = training_obj  if mode == "training" else testing_obj

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

                    # lossen constraints => possibly inaccurate but QP can sovle it more efficient (also not leading to infeasible solution)
                    newG = np.pad(G, ((0, lamb_size), (0, lamb_size)), "constant", constant_values=0)
                    newG[-lamb_size:, -lamb_size:] = -torch.eye(lamb_size)
                    newh = np.pad(h, (0, lamb_size), "constant", constant_values=-D_CONST)

                    # tighter constraints => more possible that QP cannot solve it but more accurate
                    # newG = np.pad(G, ((0, lamb_size * 2), (0, lamb_size)), "constant", constant_values=0)
                    # newG[-2*lamb_size:-lamb_size, -lamb_size:] = -torch.eye(lamb_size)
                    # newG[-lamb_size:,-lamb_size:] = torch.eye(lamb_size)
                    # newh = np.pad(h, (0, lamb_size * 2), "constant", constant_values=D_CONST)
                    # newh[-lamb_size:] = M

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
                        variance = attacker_budgets if robust_option else torch.zeros(nBatch, m_size).to(device)
                    elif robust_option:
                        variance = torch.zeros(nBatch, m_size).to(device)
                        variance[:,:-n_targets] = uncertainty_model(features).view(nBatch, m_size)[:,:-n_targets] * max_budget
                    else:
                        variance = torch.zeros(nBatch, m_size).to(device)

                    phis = torch.cat((mean, variance), dim=1).cpu()

                    def ineq_fun(x):
                        return extended_G.numpy() @ x - extended_h.numpy()
                        # return G @ x[:x_size] - h

                    def eq_fun(x):
                        return extended_A.numpy() @ x - extended_b.numpy()
                        # return A @ x[:x_size] - b

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
                        regularization_term = 0.05 * torch.eye(hess.shape[-1])
                        return hess.detach().numpy()[0] + regularization_term

                    def g_hessp(x, p):
                        x_torch = torch.Tensor(x).view(1, x_size + lamb_size)
                        p_torch = torch.Tensor(p)
                        hessp = -dual_hess.hessp(x_torch, phis, p_torch)
                        return hessp.detach().numpy()[0]

                    x0 = np.random.rand(x_size + lamb_size) # initial point
                    # x0 = np.concatenate((np.random.rand(x_size), np.zeros(lamb_size))) # initial point
                    if blackbox_option:
                        for tmp_count in range(10):
                            res = scipy.optimize.minimize(fun=g, x0=x0, method=method, jac=g_jac, hessp=g_hessp, bounds=[(0.0, 1.0)]*(x_size) + [(0.0, M)]*(lamb_size), constraints=constraints_slsqp, options={"maxiter": 100})
                            xlamb = torch.Tensor(res.x).view(1, x_size + lamb_size)

                            hess = -dual_hess.hess(xlamb, phis)
                            regularization_term = 0.05 * torch.eye(hess.shape[-1])
                            Q = hess + regularization_term

                            jac = -dual_gradient(xlamb, phis)[:,:x_size + lamb_size]
                            p = (jac.view(1, -1) - torch.matmul(xlamb, hess)).squeeze()
                    
                            # ----------------------- nus, lambs, slacks computation -------------------
                            slacks = extended_h - (extended_G @ xlamb.view(-1,1))[:,0]


                            # ------------------------------ QP function -------------------------------
                            # qp_solver = qpth.qp.QPFunction(verbose=0) # WARNING: -1 for no verbose
                            qp_solver = qpthlocal.qp.QPFunction(zhats=xlamb, nus=None, lams=None, slacks=None, verbose=True, solver=qpthlocal.qp.QPSolvers.GUROBI)

                            new_xlamb_opt = qp_solver(Q, p, extended_G, extended_h, extended_A, extended_b)

                            x0 = new_xlamb_opt.detach().numpy()
                            if abs(g(xlamb) - g(new_xlamb_opt)) < D_EPSILON:
                                # print(tmp_count)
                                break

                        if False:
                            print("old value: {}".format(g(xlamb)))
                            print("new value: {}".format(g(new_xlamb_opt)))
                            print("h - Gx: {}".format(extended_h - (extended_G @ new_xlamb_opt.view(-1,1))[0]))
                            print("b - Ax: {}".format(extended_b - (extended_A @ new_xlamb_opt.view(-1,1))[0]))

                    else: # QP direct computation
                        # Q has been precomputed
                        p = torch.cat((torch.zeros(x_size).to(device), torch.Tensor(constraint_matrix).to(device) @ mean[0] + variance[0])).to("cpu")
                        # qp_solver = qpth.qp.QPFunction(verbose=0) # WARNING: -1 for no verbose
                        qp_solver = qpthlocal.qp.QPFunction(zhats=None, nus=None, lams=None, slacks=None, verbose=True, solver=qpthlocal.qp.QPSolvers.GUROBI)
                        new_xlamb_opt = qp_solver(Q, p, extended_G, extended_h, extended_A, extended_b)
                        # TODO

                    new_x = new_xlamb_opt[:,:x_size]

                    labels_modified = constrained_attack(new_x, labels, constraint_matrix, attacker_budgets, relaxation=relaxation)
                    obj_value = (labels_modified.view(labels_modified.shape[0], 1, labels.shape[1]).to(device) @ new_x.to(device).view(*new_x.shape, 1)).mean().to(device)

                    # if torch.norm(xlamb[:,:x_size] - new_x) > D_ABNORMAL and verbose_debug:
                        # print("ABNORMAL NORM: {}".format(torch.norm(xlamb[:,:x_size] - new_x)))
                        # print(res)
                        # print(source, dest)
                        # print("mean: {}, var: {}".format(mean, variance))
                        # print("label: {}, attacker budget: {}".format(labels, attacker_budgets))
                        # print("Old xlamb")
                        # print(xlamb)
                        # print("New xlamb")
                        # print(new_xlamb_opt)
                        # print(labels_modified)

                    if torch.any(obj_value < - D_CONST):
                        print("checking...")
                        print(labels)
                        print(new_x)
                        sys.quit()

                    if robust_option:
                        loss = loss_fn(mean, labels).to(device) + loss_fn(variance, attacker_budgets).to(device)
                    else:
                        loss = loss_fn(mean, labels).to(device) + loss_fn(variance, attacker_budgets).to(device)
                        # loss = loss_fn(mean, labels_modified).to(device) + loss_fn(variance, attacker_budgets).to(device)

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

                loss_record[idx, epoch+2] = np.mean(loss_list)
                obj_record[idx, epoch+2]  = np.mean(obj_list)

    # ================================================ file processing =========================================================
    f_summary_obj.write("training obj, non-robust, two-stage, "          + ",".join([str(x) for x in training_obj[0]]) + "\n")
    f_summary_obj.write("training obj, non-robust, decision-focused, "   + ",".join([str(x) for x in training_obj[1]]) + "\n")
    f_summary_obj.write("training obj, robust, two-stage, "              + ",".join([str(x) for x in training_obj[2]]) + "\n")
    f_summary_obj.write("training obj, robust, decision-focused, "       + ",".join([str(x) for x in training_obj[3]]) + "\n")

    f_summary_obj.write("testing obj, non-robust, two-stage, "           + ",".join([str(x) for x in testing_obj[0]]) + "\n")
    f_summary_obj.write("testing obj, non-robust, decision-focused, "    + ",".join([str(x) for x in testing_obj[1]]) + "\n")
    f_summary_obj.write("testing obj, robust, two-stage, "               + ",".join([str(x) for x in testing_obj[2]]) + "\n")
    f_summary_obj.write("testing obj, robust, decision-focused, "        + ",".join([str(x) for x in testing_obj[3]]) + "\n")

    f_summary_loss.write("training loss, non-robust, two-stage, "        + ",".join([str(x) for x in training_loss[0]]) + "\n")
    f_summary_loss.write("training loss, non-robust, decision-focused, " + ",".join([str(x) for x in training_loss[1]]) + "\n")
    f_summary_loss.write("training loss, robust, two-stage, "            + ",".join([str(x) for x in training_loss[2]]) + "\n")
    f_summary_loss.write("training loss, robust, decision-focused, "     + ",".join([str(x) for x in training_loss[3]]) + "\n")

    f_summary_loss.write("testing loss, non-robust, two-stage, "         + ",".join([str(x) for x in testing_loss[0]]) + "\n")
    f_summary_loss.write("testing loss, non-robust, decision-focused, "  + ",".join([str(x) for x in testing_loss[1]]) + "\n")
    f_summary_loss.write("testing loss, robust, two-stage, "             + ",".join([str(x) for x in testing_loss[2]]) + "\n")
    f_summary_loss.write("testing loss, robust, decision-focused, "      + ",".join([str(x) for x in testing_loss[3]]) + "\n")

    f_summary_obj.close()
    f_summary_loss.close()


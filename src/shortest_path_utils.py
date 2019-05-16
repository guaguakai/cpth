import torch
import torch.nn as nn
import torch.utils.data as data_utils
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler

import networkx as nx
import numpy as np
from qpthlocal.qp import QPFunction
from qpthlocal.qp import QPSolvers
import random
import pickle

from linear import make_shortest_path_matrix

# Random Seed Initialization
SEED = random.randint(0,10000)
print("Random seed: {}".format(SEED))
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

def make_fc(num_features, num_targets, num_layers = 1, intermediate_size = 200, activation = 'relu'):
    if num_layers > 1:
        if activation == 'relu':
            activation_fn = nn.ReLU
        elif activation == 'sigmoid':
            activation_fn = nn.Sigmoid
        else:
            raise Exception('Invalid activation function: ' + str(activation))
        net_layers = [nn.Linear(num_features, intermediate_size), activation_fn()]
        for hidden in range(num_layers-2):
            net_layers.append(nn.Linear(intermediate_size, intermediate_size))
            net_layers.append(activation_fn())
        net_layers.append(nn.Linear(intermediate_size, num_targets))
        net_layers.append(nn.Sigmoid())
        # net_layers.append(nn.ReLU())
        return nn.Sequential(*net_layers)
    else:
        return nn.Sequential(nn.Linear(num_features, num_targets), nn.Sigmoid())

class Net(nn.Module):
    def __init__(self, n_features, n_targets, intermediate_size = 200):
        super(Net, self).__init__()
        self.num_features = n_features # TODO
        self.num_targets = n_targets
        self.intermediate_size = intermediate_size

        self.fc1 = nn.Linear(self.num_features, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, intermediate_size)
        self.fc3 = nn.Linear(intermediate_size, self.num_targets)

    def forward(self, x):
        x = nn.Sigmoid()(self.fc1(x))
        x = nn.Sigmoid()(self.fc2(x))
        # x = nn.Dropout(x)
        x = self.fc3(x)

        return nn.Sigmoid()(x)

def generate_graph(n_nodes=100, n_instances=300, seed=SEED):
    generation_succeed = False
    print("Generating graph...")
    while not generation_succeed:
        original_graph = nx.random_geometric_graph(n_nodes, 0.20)
        g = nx.DiGraph(original_graph)
        c = np.random.rand(g.number_of_edges())
        for idx, (u,v) in enumerate(g.edges()):
            g[u][v]['idx'] = idx
            g[u][v]['weight'] = c[idx]
        
        if nx.is_connected(original_graph):
            generation_succeed = True
        else:
            generation_succeed = False

    print("Generating sources and destinations...")
    source_list = []
    dest_list = []
    for i in range(n_instances):
        source, dest = np.random.choice(list(g.nodes()), size=2, replace=False)
        source_list.append(source)
        dest_list.append(dest)

    print("Finish generating graph!")
    return g, c, source_list, dest_list

def generate_toy_graph(n_nodes=4, n_instances=300):
    assert(n_nodes==4)
    assert(n_instances==300)
    g = nx.DiGraph()
    g.add_nodes_from([0,1,2,3])
    g.add_edges_from([(0,1), (0,2), (1,2), (2,1), (1,3), (2,3)])

    g[0][1]['idx'] = 0
    g[0][2]['idx'] = 1
    g[1][2]['idx'] = 2
    g[2][1]['idx'] = 3
    g[1][3]['idx'] = 4
    g[2][3]['idx'] = 5

    c = np.random.rand(g.number_of_edges())
    # source = 0
    # dest = 3
    source_list = np.array([0] * 100 + [0] * 100 + [2] * 100)
    # source_list = np.random.choice(list(g.nodes()), size=n_instances)
    dest_list = np.array([3] * 100 + [2] * 100 + [3] * 100)
    # dest_list = np.random.choice(list(g.nodes()), size=n_instances)
    return g, c, source_list, dest_list


def load_toy_data(args, kwargs, g, latency, n_instances, n_features=1):
    label1 = [0.3,0.2,0.0,0.0,0.3,0.2]
    label2 = [0.3,0.2,0.0,0.0,0.3,0.2]

    features = torch.Tensor([[0.5] * n_features] * n_instances)
    labels = torch.Tensor([label1] * (int(n_instances/2)) + [label2] * (n_instances - int(n_instances/2)))

    # =================== dataset spliting ======================
    dataset_size = len(features)
    train_size   = int(np.floor(dataset_size * 0.8))
    test_size    = dataset_size - train_size

    entire_dataset = data_utils.TensorDataset(features, labels, torch.arange(dataset_size))
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    train_loader = data_utils.DataLoader(entire_dataset, batch_size=args.batch_size, **kwargs, sampler=SubsetRandomSampler(train_indices))
    test_loader  = data_utils.DataLoader(entire_dataset, batch_size=args.test_batch_size, **kwargs, sampler=SubsetRandomSampler(test_indices))

    return train_loader, test_loader


def load_data(args, kwargs, g, latency, n_instances, n_constraints, n_features=500, max_budget=1.0):
    import random
    import torch
    random.seed(SEED)
    torch.manual_seed(SEED)

    n_targets = g.number_of_edges()
    true_transform = make_fc(g.number_of_edges() + n_constraints + n_targets, n_features, num_layers=5)
    
    def bimodal_random(num_samples, num_edges):
        c = torch.zeros(num_samples, num_edges)
        for i in range(num_samples):
            for j in range(num_edges):
                if random.random() < 0.2:
                    c[i,j] = random.random() + 2
                else:
                    c[i,j] = 0.3*random.random() + 0.5
        return c
    
#    c_train = torch.rand(n_train, g.number_of_edges())
#    c_test = torch.rand(n_test, g.number_of_edges())

    # =========== generating constraints with budget ============
    budgets = torch.cat((max_budget * torch.rand((n_instances, n_constraints)), -torch.zeros(n_instances, n_targets)), dim=1)
    constraint_matrix = torch.cat((torch.rand((n_constraints, n_targets)), -torch.eye(n_targets)), dim=0).numpy() # numpy matrix

    # ================== generating dataset =====================
    labels = (torch.Tensor(latency) + 0.1 * bimodal_random(n_instances, g.number_of_edges())).float()
    features = true_transform(torch.cat((labels, budgets), dim=1)).detach()

    # print(features.shape)
    # print(labels.shape)

    # =================== dataset spliting ======================
    dataset_size = len(features)
    train_size   = int(np.floor(dataset_size * 0.8))
    test_size    = dataset_size - train_size

    entire_dataset = data_utils.TensorDataset(features, labels, budgets, torch.arange(dataset_size))
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    train_loader = data_utils.DataLoader(entire_dataset, batch_size=args.batch_size, **kwargs, sampler=SubsetRandomSampler(train_indices))
    test_loader  = data_utils.DataLoader(entire_dataset, batch_size=args.test_batch_size, **kwargs, sampler=SubsetRandomSampler(test_indices))

    return train_loader, test_loader, constraint_matrix

def random_constraints(n_features, n_constraints, seed=SEED):
    np.random.seed(seed)
    max_budget = 1.0
    constraint_matrix = np.concatenate((np.random.random(size=(n_constraints, n_features)), -np.eye(n_features)), axis=0)
    budget = np.concatenate((max_budget * np.random.random(n_constraints), np.zeros(n_features)))
    return constraint_matrix, budget

class ShortestPathLoss():
    def __init__(self, n_nodes, g, c, source_list, dest_list):
        self.n_nodes = n_nodes
        self.g, self.c, self.source_list, self.dest_list = g, c, source_list, dest_list
        self.A, self.b, self.G, self.h = [], [], [], []

        assert(len(source_list) == len(dest_list)) # using the same graph g but different sources and destinations
        n_samples = len(source_list)
        for i in range(n_samples):
            tmp_A, tmp_b, tmp_G, tmp_h = make_shortest_path_matrix(self.g, self.source_list[i], self.dest_list[i])
            self.A.append(torch.Tensor(tmp_A).float())
            self.b.append(torch.Tensor(tmp_b).float())
            self.G.append(torch.Tensor(tmp_G).float())
            self.h.append(torch.Tensor(tmp_h).float())

        self.c = torch.Tensor(self.c).float()
        self.n = len(self.c)
        # self.A = torch.Tensor(self.A).float()
        # self.b = torch.Tensor(self.b).float()
        # self.G = torch.Tensor(self.G).float()
        # self.h = torch.Tensor(self.h).float()
        self.gamma = 0.1
        self.gammaQ = self.gamma * torch.eye(self.n, device="cpu")
        self.zeroQ = torch.zeros((self.n, self.n), device="cpu")

    def get_loss(self, net, features, labels, indices, eval_mode=True):
        if eval_mode:
            net.eval()

        Q = self.zeroQ
        G = self.G[indices]
        h = self.h[indices]
        A = self.A[indices]
        b = self.b[indices]

        c_pred = net(features)
        sample_number = features.shape[0]
        if c_pred.dim() == 2:
            n_train = sample_number
        else:
            n_train = 1
        c_pred = torch.Tensor.cpu(c_pred.squeeze())

        if len(A) == 0 and len(b) == 0:
            x = QPFunction(verbose=False, solver=QPSolvers.GUROBI)(Q.expand(n_train, *Q.shape), c_pred, G, h, A, b)
            # x = QPFunction(verbose=False, solver=QPSolvers.GUROBI)(Q.expand(n_train, *Q.shape), c_pred, self.G.expand(n_train, *self.G.shape), self.h.expand(n_train, *self.h.shape), self.A, self.b)
        else:
            x = QPFunction(verbose=False, solver=QPSolvers.GUROBI)(Q.expand(n_train, *Q.shape), c_pred, G, h, A, b)
            # x = QPFunction(verbose=False, solver=QPSolvers.GUROBI)(self.zeroQ.expand(n_train, *self.zeroQ.shape), c_pred, self.G.expand(n_train, *self.G.shape), self.h.expand(n_train, *self.h.shape), self.A.expand(n_train, *self.A.shape), self.b.expand(n_train, *self.b.shape))

        loss = (labels.view(sample_number, 1, labels.shape[1]).to("cpu") @ x.view(*x.shape, 1)).mean()
        net.train()
        return loss

    def relaxed_get_loss(self, net, features, labels, indices, eval_mode=True):
        if eval_mode:
            net.eval()

        Q = self.gammaQ
        G = self.G[indices]
        h = self.h[indices]
        A = self.A[indices]
        b = self.b[indices]

        c_pred = net(features)
        sample_number = features.shape[0]
        if c_pred.dim() == 2:
            n_train = sample_number
        else:
            n_train = 1
        c_pred = torch.Tensor.cpu(c_pred.squeeze())

        if len(A) == 0 and len(b) == 0:
            x = QPFunction(verbose=False, solver=QPSolvers.GUROBI)(Q.expand(n_train, *Q.shape), c_pred, G, h, A, b)
        else:
            x = QPFunction(verbose=False, solver=QPSolvers.GUROBI)(Q.expand(n_train, *Q.shape), c_pred, G, h, A, b)

        loss = (labels.view(sample_number, 1, labels.shape[1]).to("cpu") @ x.view(*x.shape, 1)).mean()
        net.train()
        return loss

    def get_two_stage_loss(self, net, features, labels, indices, eval_mode=True):
        if eval_mode:
            net.eval()

        c_pred = net(features)
        loss_fn = nn.MSELoss()
        # loss_fn = nn.BCEWithLogitsLoss() # cross entropy loss
        loss = loss_fn(c_pred, labels)
        net.train()
        return loss/len(labels)

    def get_loss_random(self, features, labels, indices):
        Q = self.zeroQ
        G = self.G[indices]
        h = self.h[indices]
        A = self.A[indices]
        b = self.b[indices]

        c_pred = torch.rand_like(labels, device="cpu")
        labels = labels.to("cpu")
        if c_pred.dim() == 2:
            n_train = features.shape[0]
        else:
            n_train = 1
        c_pred = c_pred.squeeze()

        if len(A) == 0 and len(b) == 0:
            x = QPFunction(verbose=False, solver=QPSolvers.GUROBI)(Q.expand(n_train, *Q.shape), c_pred, G, h, A, b)
        else:
            x = QPFunction(verbose=False, solver=QPSolvers.GUROBI)(Q.expand(n_train, *Q.shape), c_pred, G, h, A, b)

        loss = (labels.view(labels.shape[0], 1, labels.shape[1]) @ x.view(*x.shape, 1)).mean()
        return loss

    def get_loss_opt(self, features, labels, indices):
        Q = self.zeroQ
        G = self.G[indices]
        h = self.h[indices]
        A = self.A[indices]
        b = self.b[indices]

        labels = labels.to("cpu")
        c_pred = labels
        if c_pred.dim() == 2:
            n_train = features.shape[0]
        else:
            n_train = 1
        c_pred = c_pred.squeeze()

        if len(A) == 0 and len(b) == 0:
            x = QPFunction(verbose=False, solver=QPSolvers.GUROBI)(Q.expand(n_train, *Q.shape), c_pred, G, h, A, b)
        else:
            x = QPFunction(verbose=False, solver=QPSolvers.GUROBI)(Q.expand(n_train, *Q.shape), c_pred, G, h, A, b)

        loss = (labels.view(labels.shape[0], 1, labels.shape[1])@x.view(*x.shape, 1)).mean()
        return loss

    def get_loss_worst_case(self, net, features, labels, indices, constraint_matrix, r, eval_mode=True):
        if eval_mode:
            net.eval()

        Q = self.zeroQ
        G = self.G[indices]
        h = self.h[indices]
        A = self.A[indices]
        b = self.b[indices]

        c_pred = net(features)
        sample_number = features.shape[0]
        if c_pred.dim() == 2:
            n_train = sample_number
        else:
            n_train = 1
        c_pred = torch.Tensor.cpu(c_pred.squeeze())

        if len(A) == 0 and len(b) == 0:
            x = QPFunction(verbose=False, solver=QPSolvers.GUROBI)(Q.expand(n_train, *Q.shape), c_pred, G, h, A, b)
        else:
            x = QPFunction(verbose=False, solver=QPSolvers.GUROBI)(Q.expand(n_train, *Q.shape), c_pred, G, h, A, b)

        # print("source: {}, destination: {}, decision: {}".format(self.source_list[indices], self.dest_list[indices], x))

        # adversarial attack on the intermediate labels
        labels_modified = constrained_attack(x, labels, constraint_matrix, r)

        loss = (labels_modified.view(sample_number, 1, labels_modified.shape[1]).to("cpu") @ x.view(*x.shape, 1)).mean()
        net.train()
        return loss

    def relaxed_get_loss_worst_case(self, net, features, labels, indices, constraint_matrix, r, eval_mode=True):
        if eval_mode:
            net.eval()

        Q = self.gammaQ
        G = self.G[indices]
        h = self.h[indices]
        A = self.A[indices]
        b = self.b[indices]

        c_pred = net(features)
        sample_number = features.shape[0]
        if c_pred.dim() == 2:
            n_train = sample_number
        else:
            n_train = 1
        c_pred = torch.Tensor.cpu(c_pred.squeeze())

        if len(A) == 0 and len(b) == 0:
            x = QPFunction(verbose=False, solver=QPSolvers.GUROBI)(Q.expand(n_train, *Q.shape), c_pred, G, h, A, b)
        else:
            x = QPFunction(verbose=False, solver=QPSolvers.GUROBI)(Q.expand(n_train, *Q.shape), c_pred, G, h, A, b)

        # adversarial attack on the intermediate labels
        labels_modified = constrained_attack(x, labels, constraint_matrix, r)

        loss = (labels_modified.view(sample_number, 1, labels_modified.shape[1]).to("cpu") @ x.view(*x.shape, 1)).mean()
        net.train()
        return loss

def constrained_attack(decisions, labels, constraint_matrix, attacker_budget): # x: decision, theta: intermediate label, C: constraint matrix, r: attacker budget
    # constraint_matrix and budget r need to be concatenated with -np.eye(n_targets) and np.zeros(n_targets)
    from gurobipy import Model, GRB, LinExpr
    batch_size = len(decisions)
    assert(len(decisions) == len(labels))

    # ======================= attacker ========================
    modified_theta = torch.zeros_like(labels)
    for i in range(batch_size):
        x, theta, r = decisions[i], labels[i], attacker_budget[i].cpu().numpy()
        n = len(theta)
        m_size = len(constraint_matrix)

        model = Model()
        model.params.OutputFlag=0
        model.params.TuneOutput=0

        deltas = model.addVars(n, vtype=[GRB.CONTINUOUS]*n, lb=-theta)
        for j in range(m_size):
            model.addConstr(LinExpr(constraint_matrix[j], [deltas[k] for k in range(n)]) <= r[j])

        model.setObjective(LinExpr(x, [deltas[k] for k in range(n)]), GRB.MAXIMIZE)
        model.optimize()
        for j in range(n):
            modified_theta[i][j] = theta[j] + deltas[j].x

    # print("modifications: {}".format(modified_theta - labels))
    # print("decision: {}".format(decisions))

    return modified_theta




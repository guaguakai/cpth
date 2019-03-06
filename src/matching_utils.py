import numpy as np
import torch.nn as nn
import torch
import torch.utils.data as data_utils
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler

from qpth.qp import make_gurobi_model
from qpth.qp import QPFunction
from qpth.qp import QPSolvers

# Random Seed Initialization
SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)

def load_data(args, kwargs, labels_path="../../diff_opt/cora_graphs_bipartite.pt", features_path="../../diff_opt/cora_features_bipartite.pt"):
    labels = torch.load(labels_path)
    features = torch.load(features_path)

    # =================== truncate dataset ======================
    truncated_size = args.truncated_size
    labels = labels.view([27,50,50])[:,:truncated_size,:truncated_size]
    features = features.view([27,50,50,2866])[:,:truncated_size,:truncated_size,:]
    labels = labels.contiguous().view([27,truncated_size**2, 1])
    features = features.contiguous().view([27,truncated_size**2, 2866])

    # =================== dataset spliting ======================
    dataset_size = len(features)
    train_size   = int(np.floor(dataset_size * 0.8))
    test_size    = dataset_size - train_size

    entire_dataset = data_utils.TensorDataset(features, labels)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    train_loader = data_utils.DataLoader(entire_dataset, batch_size=args.batch_size, **kwargs, sampler=SubsetRandomSampler(train_indices))
    test_loader  = data_utils.DataLoader(entire_dataset, batch_size=args.test_batch_size, **kwargs, sampler=SubsetRandomSampler(test_indices))

    return train_loader, test_loader
    # return features, labels


def make_matching_matrix(n):
    '''
    Returns a matrix A and vector b such that x lies in the matching polytope
    for a bipartite graph with n nodes iff Ax <= b.
    '''
    lhs = list(range(n))
    rhs = list(range(n, 2*n))
    
    n_vars = len(lhs)*len(rhs)
    n_constraints = len(lhs) + len(rhs) + n_vars
    A = np.zeros((n_constraints, n_vars))
    b = np.zeros((n_constraints))
    curr_idx = 0
    edge_idx = {}
    for u in lhs:
        for v in rhs:
            edge_idx[(u,v)] = curr_idx
            curr_idx += 1
    for u in lhs:
        for v in rhs: 
            A[u, edge_idx[(u,v)]] = 1
            A[v, edge_idx[(u,v)]] = 1
            A[len(lhs)+len(rhs)+edge_idx[(u,v)], edge_idx[(u,v)]] = -1
            
    for u in lhs:
        b[u] = 1
    for u in rhs:
        b[u] = 1
    
    return A, b
    
def make_fc(num_layers, num_features, num_targets, regularizers = False, activation="relu", intermediate_size=500):
    '''
    Make a fully connected neural network with given parameters
    '''
    if num_layers > 1:
        if activation == 'relu':
            activation_fn = nn.ReLU
        elif activation == 'sigmoid':
            activation_fn = nn.Sigmoid
        else:
            raise Exception('Invalid activation function: ' + str(activation))
        if regularizers:
            net_layers = [nn.Linear(num_features, intermediate_size), nn.Dropout(), activation_fn()]
        else:
            net_layers = [nn.Linear(num_features, intermediate_size), activation_fn()]
        for hidden in range(num_layers-2):
            net_layers.append(nn.Linear(intermediate_size, intermediate_size))
            if regularizers:
    #                net_layers.append(nn.BatchNorm1d(intermediate_size))
                net_layers.append(nn.Dropout())
            net_layers.append(activation_fn())
        net_layers.append(nn.Linear(intermediate_size, num_targets))
    #        net_layers.append(nn.Sigmoid())
        return nn.Sequential(*net_layers)
    else:
        return nn.Sequential(nn.Linear(num_features, num_targets), nn.Sigmoid())

# ================================ loss functions ==============================
# data   -> features
# c_true -> true labels

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.num_features = 2866 # TODO
        self.num_targets = 1

        self.fc1 = nn.Linear(self.num_features, 500)
        self.fc2 = nn.Linear(500, self.num_targets)

    def forward(self, x):
        sample_size, expand_size, feature_size = x.shape
        x = x.view(sample_size * expand_size, -1)

        x = nn.ReLU()(self.fc1(x))
        # x = nn.Dropout(x)
        x = self.fc2(x)

        x = x.view(sample_size, expand_size, -1)
        return nn.Sigmoid()(x)

class MatchingLoss():
    def __init__(self, n_nodes):
        self.n_nodes = n_nodes
        G, h = make_matching_matrix(self.n_nodes)
        self.G = torch.from_numpy(G).float().detach()
        self.n = self.G.shape[1]
        self.h = torch.from_numpy(h).float().detach()
        self.A = torch.Tensor().detach()
        self.b = torch.Tensor().detach()
        self.zeroQ = torch.zeros((self.n, self.n), device="cpu")
        self.zeroP = torch.zeros((self.n, self.n), device="cpu")
        self.gamma = 0.1
        self.beta = 0.1
        self.gammaQ = self.gamma * torch.eye(self.n, device="cpu")
        self.betaP = self.beta * torch.eye(self.n, device="cpu")
        self.betaP_inverse = self.betaP.inverse()
        # self.betaP_inverse = np.linalg.inv(self.betaP)

        self.model_params_linear = make_gurobi_model(self.G.numpy(), self.h.numpy(), None, None, self.zeroQ.numpy())
        self.model_params_quad = make_gurobi_model(self.G.numpy(), self.h.numpy(), None, None, self.gammaQ.numpy())

        self.expanded_G = F.pad(self.G, (0,self.n,0,self.n), "constant", 0)
        self.expanded_G[-self.n:, -self.n:] = -torch.eye(self.G.shape[1])
        # self.expanded_A = F.pad(self.A, (0,0,0,self.n), "constant", 0)
        self.expanded_h = F.pad(self.h, (0,self.n), "constant", 0)

        P_term = torch.cat((torch.cat((self.betaP_inverse, -self.betaP_inverse), dim=0), torch.cat((-self.betaP_inverse, self.betaP_inverse), dim=0)), dim=1)
        self.expanded_gammaQ = F.pad(self.gammaQ, (0,self.n,0,self.n), "constant", 0) + P_term
        self.expanded_zeroQ =  F.pad(self.zeroQ,  (0,self.n,0,self.n), "constant", 0) + P_term

        self.model_robust_params_linear = make_gurobi_model(self.expanded_G.numpy(), self.expanded_h.numpy(), None, None, self.expanded_zeroQ.numpy())
        self.model_robust_params_quad   = make_gurobi_model(self.expanded_G.numpy(), self.expanded_h.numpy(), None, None, self.expanded_gammaQ.numpy())

    def get_loss(self, net, features, labels, eval_mode=True):
        if eval_mode:
            net.eval()

        Q = self.zeroQ
        model_params = self.model_params_linear

        c_pred = -net(features)
        sample_number = features.shape[0]
        if c_pred.dim() == 3:
            n_train = sample_number
        else:
            n_train = 1
        c_pred = torch.Tensor.cpu(c_pred.squeeze())

        if len(self.A) == 0 and len(self.b) == 0:
            x = QPFunction(verbose=False, solver=QPSolvers.GUROBI, model_params=model_params)(Q.expand(n_train, *Q.shape), c_pred, self.G.expand(n_train, *self.G.shape), self.h.expand(n_train, *self.h.shape), self.A, self.b)
        else:
            x = QPFunction(verbose=False, solver=QPSolvers.GUROBI, model_params=model_params)(Q.expand(n_train, *Q.shape), c_pred, self.G.expand(n_train, *self.G.shape), self.h.expand(n_train, *self.h.shape), self.A.expand(n_train, *self.A.shape), self.b.expand(n_train, *self.b.shape))

        loss = (labels.view(sample_number, 1, labels.shape[1]).to("cpu") @ x.view(*x.shape, 1)).mean()
        net.train()
        return loss

    def relaxed_get_loss(self, net, features, labels, eval_mode=True):
        if eval_mode:
            net.eval()

        Q = self.gammaQ
        model_params = self.model_params_quad

        c_pred = -net(features)
        sample_number = features.shape[0]
        if c_pred.dim() == 3:
            n_train = sample_number
        else:
            n_train = 1
        c_pred = torch.Tensor.cpu(c_pred.squeeze())

        if len(self.A) == 0 and len(self.b) == 0:
            x = QPFunction(verbose=False, solver=QPSolvers.GUROBI, model_params=model_params)(Q.expand(n_train, *Q.shape), c_pred, self.G.expand(n_train, *self.G.shape), self.h.expand(n_train, *self.h.shape), self.A, self.b)
        else:
            x = QPFunction(verbose=False, solver=QPSolvers.GUROBI, model_params=model_params)(Q.expand(n_train, *Q.shape), c_pred, self.G.expand(n_train, *self.G.shape), self.h.expand(n_train, *self.h.shape), self.A.expand(n_train, *self.A.shape), self.b.expand(n_train, *self.b.shape))

        loss = (labels.view(sample_number, 1, labels.shape[1]).to("cpu") @ x.view(*x.shape, 1)).mean()
        net.train()
        return loss

    def get_robust_loss(self, net, features, labels, eval_mode=True):
        # TODO
        if eval_mode:
             net.eval()
        Q = self.expanded_zeroQ
        G = self.expanded_G
        h = self.expanded_h
        model_params = self.model_robust_params_linear
        c_pred = -net(features)
        sample_number = features.shape[0]
        if c_pred.dim() == 3:
            n_train = sample_number
        else:
            n_train = 1
        c_pred = F.pad(torch.Tensor.cpu(c_pred.squeeze()), (self.n,0), "constant", 0)

        if len(self.A) == 0 and len(self.b) == 0:
            x = QPFunction(verbose=False, solver=QPSolvers.GUROBI, model_params=model_params)(Q.expand(n_train, *Q.shape), c_pred, G.expand(n_train, *G.shape), h.expand(n_train, *h.shape), self.A, self.b)
        else:
            x = QPFunction(verbose=False, solver=QPSolvers.GUROBI, model_params=model_params)(Q.expand(n_train, *Q.shape), c_pred, G.expand(n_train, *G.shape), h.expand(n_train, *h.shape), self.A.expand(n_train, *self.A.shape), self.b.expand(n_train, *self.b.shape))

        loss = (labels.view(sample_number, 1, labels.shape[1]).to("cpu") @ ((x.view(*x.shape, 1))[:,:self.n,:])).mean()
        net.train()
        return loss

    def relaxed_get_robust_loss(self, net, features, labels, eval_mode=True):
        # TODO
        if eval_mode:
             net.eval()
        Q = self.expanded_gammaQ
        G = self.expanded_G
        h = self.expanded_h
        model_params = self.model_robust_params_quad
        c_pred = -net(features)
        sample_number = features.shape[0]
        if c_pred.dim() == 3:
            n_train = sample_number
        else:
            n_train = 1
        c_pred = F.pad(torch.Tensor.cpu(c_pred.squeeze()), (self.n,0), "constant", 0)

        if len(self.A) == 0 and len(self.b) == 0:
            x = QPFunction(verbose=False, solver=QPSolvers.GUROBI, model_params=model_params)(Q.expand(n_train, *Q.shape), c_pred, G.expand(n_train, *G.shape), h.expand(n_train, *h.shape), self.A, self.b)
        else:
            x = QPFunction(verbose=False, solver=QPSolvers.GUROBI, model_params=model_params)(Q.expand(n_train, *Q.shape), c_pred, G.expand(n_train, *G.shape), h.expand(n_train, *h.shape), self.A.expand(n_train, *self.A.shape), self.b.expand(n_train, *self.b.shape))

        loss = (labels.view(sample_number, 1, labels.shape[1]).to("cpu") @ ((x.view(*x.shape, 1))[:,:self.n,:])).mean()
        net.train()
        return loss

    def get_two_stage_loss(self, net, features, labels, eval_mode=True):
        if eval_mode:
            net.eval()

        c_pred = net(features)
        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(c_pred, labels)
        net.train()
        return loss/len(labels)

    def get_loss_random(self, features, labels):
        c_pred = -torch.rand_like(labels, device="cpu")
        labels = labels.to("cpu")
        if c_pred.dim() == 3:
            n_train = features.shape[0]
        else:
            n_train = 1
        c_pred = c_pred.squeeze()
        if len(self.A) == 0 and len(self.b) == 0:
            x = QPFunction(verbose=False, solver=QPSolvers.GUROBI, model_params=self.model_params_linear)(self.zeroQ.expand(n_train, *self.zeroQ.shape), c_pred, self.G.expand(n_train, *self.G.shape), self.h.expand(n_train, *self.h.shape), self.A, self.b)
        else:
            x = QPFunction(verbose=False, solver=QPSolvers.GUROBI, model_params=self.model_params_linear)(self.zeroQ.expand(n_train, *self.zeroQ.shape), c_pred, self.G.expand(n_train, *self.G.shape), self.h.expand(n_train, *self.h.shape), self.A.expand(n_train, *self.A.shape), self.b.expand(n_train, *self.b.shape))
        loss = (labels.view(labels.shape[0], 1, labels.shape[1]) @ x.view(*x.shape, 1)).mean()
        return loss
    
    def get_loss_opt(self, features, labels):
        labels = labels.to("cpu")
        c_pred = -labels
        if c_pred.dim() == 3:
            n_train = features.shape[0]
        else:
            n_train = 1
        c_pred = c_pred.squeeze()
        if len(self.A) == 0 and len(self.b) == 0:
            x = QPFunction(verbose=False, solver=QPSolvers.GUROBI, model_params=self.model_params_linear)(self.zeroQ.expand(n_train, *self.zeroQ.shape), c_pred, self.G.expand(n_train, *self.G.shape), self.h.expand(n_train, *self.h.shape), self.A, self.b)
        else:
            x = QPFunction(verbose=False, solver=QPSolvers.GUROBI, model_params=self.model_params_linear)(self.zeroQ.expand(n_train, *self.zeroQ.shape), c_pred, self.G.expand(n_train, *self.G.shape), self.h.expand(n_train, *self.h.shape), self.A.expand(n_train, *self.A.shape), self.b.expand(n_train, *self.b.shape))
        loss = (labels.view(labels.shape[0], 1, labels.shape[1])@x.view(*x.shape, 1)).mean()
        return loss


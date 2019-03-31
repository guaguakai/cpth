import torch
import torch.nn as nn
from qpth.qp import QPFunction
from qpth.qp import QPSolvers
import numpy as np
import networkx as nx

def make_shortest_path_matrix(g, s, t):
    '''
    For a given graph g, produce a constraint matrix for the flow 
    polytope with one unit of flow going from s to t.
    '''
    import sympy
    A = np.zeros((g.number_of_nodes()+2, g.number_of_edges()))
    b = np.zeros((g.number_of_nodes())+2)
    G = np.zeros((g.number_of_edges(), g.number_of_edges()))
    h = np.zeros((g.number_of_edges()))
    #flow conservation constraints
    for v in g:
        if v != s and v != t:
            for u in g.predecessors(v):
                A[v, g[u][v]['idx']] = 1
            for u in g.successors(v):
                A[v, g[v][u]['idx']] = -1
            b[v] = 0
        elif v == s:
            for u in g.successors(v):
                A[v, g[v][u]['idx']] = 1
            b[v] = 1
        elif v == t:
            for u in g.predecessors(v):
                A[v, g[u][v]['idx']] = 1
            b[v] = 1
    for u in g.predecessors(s):
        A[len(g), g[u][s]['idx']] = 1
    for u in g.successors(t):
        A[len(g) + 1, g[t][u]['idx']] = 1
    #non-negativity constraints
    for u,v in g.edges():
        G[g[u][v]['idx'], g[u][v]['idx']] = -1
    
    #check for linearly dependent rows and remove them
    _, inds = sympy.Matrix(A).T.rref()
    A = A[np.array(inds)]
    b = b[np.array(inds)]
        
    return A, b, G, h

def make_gurobi(c, A, b):
    import gurobipy as gp
    m = gp.Model()
    m.params.OutputFlag = 0
    variables = []
    for i in range(len(c)):
        variables.append(m.addVar(lb = 0, obj = c[i]))
    m.update()
    for i in range(A.shape[0]):
        var_nonzero = np.where(A[i] != 0)[0]
        m.addConstr(gp.quicksum(variables[j]*A[i,j] for j in var_nonzero) == b[i])
    m.update()
    return m, variables

def make_gurobi_quadratic(c, A, b, mu):
    import gurobipy as gp
    m = gp.Model()
    m.params.OutputFlag = 0
    variables = []
    for i in range(len(c)):
        variables.append(m.addVar(lb = -gp.GRB.INFINITY))
    m.update()
    constraints = []
    for i in range(A.shape[0]):
        var_nonzero = np.where(A[i] != 0)[0]
        constraints.append(m.addConstr(gp.quicksum(variables[j]*A[i,j] for j in var_nonzero) == b[i]))
    nonnegative_constraints = []
    for i in range(len(c)):
        nonnegative_constraints.append(m.addConstr(variables[i] >= 0))
    m.setObjective(gp.quicksum(variables[i]*c[i] for i in range(len(c))) + mu*gp.quicksum(variables[i]*variables[i] for i in range(len(c))))
    m.update()
    return m, variables, constraints, nonnegative_constraints

        
            
def frank_wolfe(c, A, b, mu):
    import gurobipy as gp
    x = np.random.rand(len(c))
    m, edge_vars = make_gurobi(c, A, b)
    gamma = 0.1
    for t in range(100):
        grad = c + 2*mu*x
        m.setObjective(gp.quicksum(grad[i]*edge_vars[i] for i in range(len(grad))))
        m.optimize()
        new_x = np.array([edge_vars[i].x for i in range(len(grad))])
#        print(((x - new_x)**2).mean())
        x = (1 - gamma)*x + gamma*new_x
    return x


def primal_dual(c, A, b, mu):
    def lagrangian(x, c, A, b, mu, lam, nu):
        return c.t()@x + mu*torch.norm(x) - lam.t()@x + nu.t()@(A@x - b)
    nu = torch.rand(A.shape[0], 1, requires_grad=True)
    lam = torch.rand_like(c, requires_grad=True)
    x = torch.rand_like(c, requires_grad=True)
    optim_dual = torch.optim.SGD([lam, nu], nesterov=True, momentum = 0.9, lr=0.001)
    for t_dual in range(200):
        optim_primal = torch.optim.SGD([x], nesterov=True, momentum=0.9, lr=0.001)
        for t_primal in range(200):
            loss = lagrangian(x, c, A, b, mu, lam, nu)
            optim_primal.zero_grad()
            loss.backward()
            optim_primal.step()
        loss = -lagrangian(x, c, A, b, mu, lam, nu)
        optim_dual.zero_grad()
        loss.backward()
        optim_dual.step()
        lam.data[lam < 0] = 0
    return x, lam, nu

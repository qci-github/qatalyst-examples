import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from itertools import permutations

def plot_graph_plain(graph, node_color='r'):
    """
    Plot a NetworkX graph with no special not assignments.
    """
    fig = plt.figure(figsize=(10,8))
    ax = fig.gca()
    pos = nx.spring_layout(graph)
    nx.draw_networkx_nodes(graph, pos, node_color=node_color, node_size=200)
    nx.draw_networkx_edges(graph, pos)
    
    
    
def plot_graph_classes(graph, nodes, num_classes, node_colors=['r', 'b', 'g', 'c']):
    """
    Plot up to 4 classes (see node_colors)
    """
    fig = plt.figure(figsize=(10,8))
    ax = fig.gca()
    pos = nx.spring_layout(graph)
    for k in range(num_classes):
        nodelist = [node['id'] for node in nodes if node['class'] == k]
        nx.draw_networkx_nodes(graph, pos, nodelist=nodelist, node_color=node_colors[k])
    
    nx.draw_networkx_edges(graph, pos)
    ax.set_title(f'Graph with {num_classes} classes')
    
    
def plot_qap(soln, node_colors=['g', 'y']):
    """
    soln = array([0, 2, 1, 3, 4]) refers to locations and indices of the soln array
        refer to the facilities.
        
    Ref. https://stackoverflow.com/a/62519225/4001108
    """    
    B = nx.Graph()

    # facilities
    facilities = [str(x) for x in range(len(soln))]
    B.add_nodes_from(facilities, bipartite=0) # Add the node attribute "bipartite"
    
    # locations
    B.add_nodes_from(soln, bipartite=1)
    
    B.add_edges_from([(u,soln[int(u)]) for u in facilities])
    
    pos = {}
    pos.update((node, (1, index)) for index, node in enumerate(facilities))
    pos.update((node, (2, index)) for index, node in enumerate(soln))
    pos = nx.bipartite_layout(B, facilities)
    
    top_color = [node_colors[0] for _ in facilities]
    bottom_color = [node_colors[1] for _ in facilities]
    colors = top_color + bottom_color
    nx.draw(B, pos=pos, with_labels=True, node_color=colors, node_size=300)
    plt.title(r"Optimal assignment of Facilities $\leftrightarrow$Locations")
    plt.show()

def assignment_from_solution(solution, n):
    assignment = []
    for i in range(n):
        base_idx = 5*i
        for j in range(n):
            if solution[base_idx+j] > 0:
                assignment.append(j)
    assert len(assignment) == n, "Invalid solution, incorrect number of assignments"
    assert set(assignment) == set(range(n)), "Invalid solution, duplicated assignemnts"
    return assignment

def create_qap_objective(A, B, C, n, num_variables):
    objective_data = []
    objective = dict()
    for i in range(n):
        for k in range(n):
            for j in range(n):
                for l in range(n):
                    # entry = {"i": i*n+k, "j": j*n+l}
                    key1 = (i*n+k, j*n+l)
                    key2 = (j*n+l, i*n+k)
                    if i == j and k == l:
                        objective[key1] = objective.get(key1, 0) + float(C[i, k])
                    else:
                        objective[key1] = objective.get(key1, 0) + float(A[i, j] * B[k, l] / 2)
                        objective[key2] = objective[key1]
    objective_data = [{"i": i, "j": j, "val": val} for (i, j), val in objective.items() if val != 0.0]
    objective_file = {"file_type": "objective", "file_name": "qap_01_obj.json", 
                    "data": objective_data, "num_variables": num_variables}
    return objective_file

def create_qap_constraints(n):
    constraints = np.zeros((2*n, n**2))
    for i in range(n):
        for j in range(n):
            constraints[i, i * n + j] = 1
            constraints[n + i, i + n * j] = 1
    rhs = np.ones((2*n,))
    constraint_data = []
    for i in range(constraints.shape[0]):
        for j in range(constraints.shape[1]):
            if constraints[i, j] != 0:
                constraint_data.append({"i": i, "j": j, "val": float(constraints[i, j])})
    rhs_data = rhs.tolist()

    return constraint_data, rhs_data

def find_index_of_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def gen_feasible_solutions(n):
    for p in permutations(range(n)):
        sol = np.zeros(n*n)
        for i, p in enumerate(p):
            sol[i*n+p] = 1
        yield sol


def convert_qubo_to_ising(qubo: np.ndarray = None, triu: bool = True):
    """
    Helper function for objective_qaoa
    Args:
      qubo: a symmetric array.
      triu: bool indicating upper triangular or symmetric
    Returns:
      J: an upper triangular or symmetric array with same shape as qubo for Ising quadratic coupling terms.
      h: a row array for the linear biases.
      offset: float.
    Correctness proof:
    from sympy import *
    q = IndexedBase(“q”)
    x = IndexedBase(“x”)
    n = 3
    QQ = np.zeros((n, n), dtype=object)
    for i in range(n):
      for j in range(n):
    QQ[i, j] = q[i, j]
    QQ[j, i] = q[i, j]
    bigx = np.array([x[i] for i in range(n)])
    bigs = bigs = 2*bigx -1
    t1 = expand(np.matmul(bigx, np.matmul(QQ, bigx)))
    for i in range(QQ.shape[0]):
      t1= t1.subs(x[i]**2, x[i])
    JJ, hh, offf = convert_qubo_to_ising(QQ)
    t2 = expand(np.matmul(bigs, np.matmul(JJ, bigs.T))) + expand(np.matmul(hh, bigs.T)) + offf
    t1-t2==0
    """
    n = qubo.shape[0]
    h = ((qubo @ np.ones((n, 1))) / 2).flatten()
    offset = ((np.ones((n, 1)).T @ (qubo @ np.ones((n, 1)))) / 4)[
      0, 0
    ] + qubo.trace() / 4
    # diagonal part contributes to the offset only
    Q = qubo - np.diagflat(qubo.diagonal())
    J = Q / 4
    if triu == True:
      J = 2 * np.triu(J)
    # shape of h is (n,) and J is (n,n)
    ising = np.append(np.expand_dims(h, 1), J, axis=1)
    return ising, offset

def normalize_grid(sample, grid_count):
    sample = np.array(sample, dtype=np.float64)
    sample = 64*sample/np.max(sample)
    sample[:2*grid_count] *= float(grid_count) / np.sum(sample[:2*grid_count])
    sample[2*grid_count:] *= float(100-grid_count) / np.sum(sample[2*grid_count:])
    return sample

def extract_solution(sample, grid_count):
    sample = normalize_grid(sample, grid_count)
    grid_selection = np.where(sample[:grid_count]>0.5)
    power_values = sample[grid_count*2:]
    return grid_selection, power_values
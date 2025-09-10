import sys 
import dimod
import time
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import dwave_networkx as dnx
import dwave.inspector
import dwavebinarycsp
import pandas as pd
import itertools
from itertools import permutations 
import operator
from operator import itemgetter
from dimod.reference.samplers import ExactSolver
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
from dwave.system import DWaveSampler, EmbeddingComposite
from dwave_networkx.utils import binary_quadratic_model_sampler
from dwave.system import TilingComposite
from minorminer import find_embedding
from scipy.stats import binom
import networkx as nx
import dwave_networkx as dnx
import pyqubo
from pylab import *
from scipy.integrate import odeint
from scipy.optimize import brentq
from dwave.embedding.chain_strength import scaled
from dwave.system.samplers import DWaveSampler
from sys import maxsize
%matplotlib inline

def create_cities(N):
    cities = []
    for i in range(N):
        cities.append(np.random.rand(1)*10)
    return np.array(cities)


def distance_between_points(point_A, point_B):
    return np.sqrt((point_A[0] - point_B[0])**2)
    # + (point_A[1] - point_B[1])**2)
    

def get_distance_matrix(cities):
    number_of_cities = len(cities)
    matrix = np.zeros((number_of_cities, number_of_cities))
    for i in range(number_of_cities):
        for j in range(i, number_of_cities):
            matrix[i][j] = distance_between_points(cities[i], cities[j])
            matrix[j][i] = matrix[i][j]
    return matrix

cities = create_cities(10)
distance_matrix = get_distance_matrix(cities)

min = sys.maxsize

for i in range(len(distance_matrix)):
    for j in range(len(distance_matrix)):
        if distance_matrix[i][j] < min:
            min = i
            max = min
        if distance_matrix[i][j] > min:
            max = i
for i in range(len(distance_matrix)):
    for j in range(len(distance_matrix)):
        distance_matrix[i][j] = (distance_matrix[i][j] - min)/(max - min)
print(min, max)
print(distance_matrix)
    
    
def index(i, j, n):
    if i == j:
        raise ValueError
    elif i > j:
        return index(j, i, n)
    else:
        return int(i*n - i*(i+1)/2 + j - (i+1))

def build_constraint_matrix(n):
    m = int(n*(n-1)/2)
    C = np.zeros((m,m))
    for i in range(0, n):
        for j in range(0, n):
            if i == j:
                continue
            k = index(i, j, n)
            C[k,k] += 2
        for a in range(0, n):
            for b in range(0, n):
                if a == b or a == i or b == i:
                    continue
                ia = index(i,a,n)
                ib = index(i,b,n)
                C[ia,ib] += 1
    return C

def build_objective_matrix(M):
    n, _ = M.shape
    # m is the total # of binary variables we have to solve for
    # basically given any two nodes, need to decide if there is an edge connecting them (a binary variable)
    m = int(n*(n-1)/2)
    Q = np.zeros((m,m))
    k = 0
    for i in range(0, n):
        for j in range(i+1, n):
            #M[i,j] + M[j,i] is the cost to travel from i to j (or vice-versa)
            Q[k, k] = (M[i,j] + M[j,i])
            k += 1
    # diagonal matrix of biases
    return Q

def is_valid_solution(X):
    rows, cols = X.shape
    for i in range(rows):
        count = 0
        for j in range(cols):
            if X[i,j] == 1:
                count += 1
        if not count == 2:
            return False
    return True
    
def build_solution(sample):
    n, _ = M.shape # this will use the global M variable
    m = len(sample)
    assert m == int(n*(n-1)/2)
    X = np.zeros((n,n))
    k = 0
    for i in range(n):
        for j in range(i+1, n):
            X[i,j] = X[j,i] = sample[k]
            k += 1
    return X

def score(M, X):
    return np.sum(np.multiply(M, X))    


out_file = 'output.txt'
num_samples = 100

# the matrix of paiwise costs (cost to travel from node i to node j). this need not be a symmetric matrix but the diagonal entries are ignored
# and assumed to be zero (don't care)
M = distance_matrix
Q = build_objective_matrix(M)
lagrange_multiplier = np.max(np.abs(M))

# now we just need to add the constraint that each city is connected to exactly 2 other cities
# we do this using the method of lagrange multipliers where the constraint is absorbed into the objective function
# this is the hardest part of the problem
n, _ = M.shape
C = build_constraint_matrix(n)
# print(C)

qubo = Q + lagrange_multiplier * C
quboed = nx.from_numpy_matrix(qubo)
sampler = EmbeddingComposite(DWaveSampler(endpoint='https://cloud.dwavesys.com/sapi',        token='DEV-d9500e99062cdd39d17bc2f4a0993eb1cae4e053', 
solver='Advantage_system6.1')) # QPU sampler to run in production
t0 = time.perf_counter()
sampleset = sampler.sample_qubo(qubo, num_reads=num_samples, chain_strength=scaled)
t1 = time.perf_counter()
dwave.inspector.show(sampleset)
have_solution = False
problem_id = sampleset.info['problem_id']
chain_strength = sampleset.info['embedding_context']['chain_strength']
with open(out_file, 'w') as f:
    f.write(f"Problem Id: {problem_id}\n")        # does not depend on sample  
    count = 0
    for e in sampleset.data(sorted_by='energy'):
        sample = e.sample
        energy = e.energy
        num_occurrences = e.num_occurrences
        chain_break_fraction = e.chain_break_fraction
        X = build_solution(sample)
        if is_valid_solution(X):
            have_solution = True
            score = score(M, X)
            f.write(f"Solution:\n")
            f.write(f"{X}\n")
            f.write(f"Score: {score}\n")
            f.write(f"{sample}\n")
            f.write(f"index: {count}\n")
            f.write(f"energy: {energy}\n")
            f.write(f"num_occurrences: {num_occurrences}\n")            
            f.write(f"chain break fraction: {chain_break_fraction}\n")            
            break   # break out of for loop
        count += 1
    f.write(f"chain strength: {chain_strength}\n")  # does not depend on sample
    f.write(f"lagrange multiplier: {lagrange_multiplier}\n")
    f.write(f"Time: {t1-t0:0.4f} s\n")
    if not have_solution:
        # https://docs.ocean.dwavesys.com/en/latest/examples/inspector_graph_partitioning.html
        # this is the overall chain break fraction
        chain_break_fraction = np.sum(sampleset.record.chain_break_fraction)/num_samples
        f.write("did not find any solution\n")
        f.write(f"chain break fraction: {chain_break_fraction}\n")
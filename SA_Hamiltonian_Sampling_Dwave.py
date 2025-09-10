import Simulated_annealing as tl
import sys 
import dimod
import time
import csv
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import dwave_networkx as dnx
import dwave.inspector 
import dwavebinarycsp
import pandas as pd
import itertools
import seaborn as sns
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

np.random.seed(1000)
N = n*n
iterations = 10*N
A, B, C = 1, 5, 5
beta = 0.5

c = tl.cities(n)

c.cities_initialize()

c.get_h(A, B, C)
c.get_J(A, B, C)

c.spins_initialize()

linear_constraint = np.diag(c.h)
Hamiltonian = c.J + linear_constraint
sampler = EmbeddingComposite(DWaveSampler(endpoint='https://cloud.dwavesys.com/sapi', token='enter_access_token', 
    solver='Advantage2_prototype2.3')) # QPU sampler to run in production

sampleset = sampler.sample_ising(c.h, c.J, num_reads=1000, chain_strength=scaled)
time = sampleset.info['timing']['qpu_access_time']
energy = c.simulated_annealing(iterations, beta, annealer=tl.afunc_log, autostop=0.05, timer=True)

df = pd.DataFrame(data=sampleset)
print(df.iloc[1])

soln = np.array(df.iloc[3]).reshape(n, n)
print(soln)

sns.heatmap(soln)
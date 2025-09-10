# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import time

class cities():
    
    def __init__(self, num_cities):
        self.n = num_cities
        self.N = num_cities*num_cities
        
    def cities_initialize(self, x_range=[0, 1], y_range=[0, 1]):
        self.coords = np.column_stack((np.random.uniform(x_range[0], x_range[1], self.n),
                                 np.random.uniform(y_range[0], y_range[1], self.n)))
        self.W = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(i):
                self.W[i, j] = np.linalg.norm(self.coords[i] - self.coords[j])
                self.W[j, i] = self.W[i, j]
    
    def cities_set_coords(self, coordinates):
        self.coords = coordinates
        self.W = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(i):
                self.W[i, j] = np.linalg.norm(self.coords[i] - self.coords[j])
                self.W[j, i] = self.W[i, j]
    
    def spins_initialize(self):
        self.spins = np.random.choice([-1, 1], self.N)
        self.spins_copy = self.spins.copy()
        
    def spins_set_initial(self, spins_in):
        self.spins = spins_in.copy().flatten()
        self.spins_copy = self.spins.copy()
        
    def get_J(self, A, B, C):
        ja = np.eye(self.n, k=-1) + np.eye(self.n, k=1)
        ja[0, self.n-1] += 1
        ja[self.n-1, 0] += 1
        JA = np.kron(ja, self.W)
        
        jb = np.ones((self.n, self.n)) - np.eye(self.n)
        JB = np.kron(np.eye(self.n), jb)
        
        jc = np.ones((self.n, self.n)) - np.eye(self.n)
        JC = np.kron(jc, np.eye(self.n))
        
        self.J = 0.25*A*JA + 0.25*B*JB + 0.25*C*JC
    
    def get_h(self, A, B, C):
        ha = np.sum(np.kron(np.ones(self.n), self.W), axis=0)
        
        hb = (self.n-2)*np.ones(self.n*self.n)
        
        hc = (self.n-2)*np.ones(self.n*self.n)
        
        self.h = 0.5*A*ha + 0.5*B*hb + 0.5*C*hc
        
    def get_energy(self):
        return calc_energy(self)
    
    def simulated_anealing(self, max_steps, beta, **kwargs):
        E_list = []
        E_list.append(self.get_energy())
        
        if 'annealer' in kwargs:
            annealer_func = kwargs['annealer']
        else:
            annealer_func = no_anneal
            
        ti = time.time()
            
        for i in range(1, max_steps+1):
            E_list.append(annealing_step(self, beta*annealer_func(i), E_list[i-1]))
            
            if ('autostop' in kwargs) and (i%self.N==0):
                ac = auto_correlation(self)
                if ac <= kwargs['autostop']:
                    break
                else:
                    self.spins_copy = self.spins.copy()
        
        tf = time.time()
        
        E_list = np.asarray(E_list)
        if ('timer' in kwargs) and (kwargs['timer']==True):
            print(tf-ti)
        return E_list / (self.N*self.N)
    
def calc_energy(cities):
    return np.dot(cities.spins, np.dot(cities.J, cities.spins) + cities.h)

def annealing_step(cities, beta, E_in):
    site = np.random.choice(cities.N)
    cities.spins[site] *= -1
    new_E = cities.get_energy()
    dE = new_E - E_in
    flip_prob = np.exp(-beta * dE)
    if dE <= 0 or flip_prob > np.random.random():
        E_out = new_E
    else:
        cities.spins[site] *= -1
        E_out = E_in
    return E_out

def no_anneal(i):
    return 1

def afunc_log(i):
    return np.log(1+i)

def afunc_power(i, r=0.5):
    return np.power(r, 1-i)

def auto_correlation(cities):
    return np.abs(cities.spins-cities.spins_copy).sum() / (2*cities.N)


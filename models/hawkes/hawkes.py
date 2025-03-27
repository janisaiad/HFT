import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import polars as pl
import pandas as pd
from scipy.integrate import quad

class Hawkes: # N dimensionnal hawkes process
    def __init__(self, phi: callable, psi: callable, mu: float, N: int):
        # phi returns a N dimensionnal vector
        # psi returns a N dimensionnal vector
        # mu is a N dimensionnal vector
        self.dim = N
        self.phi = phi 
        self.psi = psi
        self.mu = mu
        self.l1_norm_phi = None
        self.list_of_events = [[] for _ in range(N)] # matrix of time events, of size N x T
        
        
        
    def add_event(self, event: float, dimension: int):
        self.list_of_events[dimension].append(event)
    
   
   
   
    
    # EVENTS
    def print_events(self):
        for i in range(self.dim):
            print(self.list_of_events[i])
        return self.list_of_events
    
    def get_events(self, t: float) -> np.ndarray:
        return self.list_of_events[t]
    
    def verify_l1_norm_phi(self) -> bool:
        l1_matrix = np.zeros((self.dim, self.dim))
        for i in range(self.dim):
            for j in range(self.dim):
                def integrand(t):
                    return np.abs(self.phi(t)[i,j])
                l1_matrix[i,j], _ = quad(integrand, 0, np.inf) # this is managed by scipy
        spectral_radius = np.max(np.abs(np.linalg.eigvals(l1_matrix)))
        self.l1_norm_phi = l1_matrix
        return spectral_radius < 1
    
    
    
    # L1 NORM
    
    def print_l1_norm_phi(self):
        l1_matrix = np.zeros((self.dim, self.dim))
        for i in range(self.dim):
            for j in range(self.dim):
                l1_matrix[i,j] = quad(lambda t: np.abs(self.phi(t)[i,j]), 0, np.inf)[0]
        print(l1_matrix)
        self.l1_norm_phi = l1_matrix
        return l1_matrix
    
    def get_l1_norm_phi(self):
        l1_matrix = np.zeros((self.dim, self.dim))
        for i in range(self.dim):
            for j in range(self.dim):
                l1_matrix[i,j] = quad(lambda t: np.abs(self.phi(t)[i,j]), 0, np.inf)[0]
        self.l1_norm_phi = l1_matrix
        return l1_matrix
    
    
    # INTENSITIES
    def print_intensity(self, t: float):
        intensity = self.get_intensity(t)
        print(intensity)
        return intensity

    def get_intensity(self, t: float) -> np.ndarray: # returns a vector of intensities
        intensity = self.mu.copy()
        for dim in range(self.dim):
            for event in self.list_of_events[dim]:
                intensity += self.phi(t - event)[:, dim]  # phi returns a matrix, take column dim
        return intensity
    
    
    
    
    def get_average_intensity(self, t: float) -> float:
        mean_vector = np.zeros(self.dim)
        if self.l1_norm_phi is None:
            # Get L1 norm matrix
            l1_matrix = self.get_l1_norm_phi()
            # Calculate (I-L)^-1 * mu
            I = np.eye(self.dim)  # Identity matrix
            mean_vector = np.linalg.inv(I - l1_matrix) @ self.mu  # Matrix multiplication with inverse
            return mean_vector
        
        return (np.eye(self.dim) - self.l1_norm_phi) ** -1 @ self.mu
    
    
   
    def simulate(self, T: int, n: int):
        pass



    def fit(self, X: np.ndarray): # X is a numpy array of events
        pass


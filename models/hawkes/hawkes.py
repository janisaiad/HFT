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
        self.convolution_threshold = 10 # number of time we convolve phi with itself
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
    
    
    
    
    # here we compute the equation (4) of the pape Bacry et al
    def get_average_intensity(self, t: float) -> float:
        mean_vector = np.zeros(self.dim)
        I = np.eye(self.dim)
        if self.l1_norm_phi is None:
            l1_matrix = self.get_l1_norm_phi()
            mean_vector = np.linalg.inv(I - l1_matrix) @ self.mu 
            return mean_vector
        return np.linalg.inv(I - self.l1_norm_phi) @ self.mu
    
    
    
    
    
    # part 2.2 of Bacry et al
    
    # convolution product utils, to be put in utils folder after
    def convolution_product(self, function: callable) -> callable: # this is the convolution product of
        return lambda t: quad(lambda tau: function(t - tau) * function(tau), 0, t)[0] # we return a function of t
    
    
    def convolution_product_matrix(self, function: callable) -> callable:
        
        def result(t)->np.ndarray:
            psi_matrix = np.zeros((self.dim, self.dim))
            for i in range(self.dim):
               for j in range(self.dim):
                   function_to_apply = lambda tau: function(tau)[i,j]
                   psi_matrix[i,j] = self.convolution_product(function_to_apply)(t)
            return psi_matrix
        
        return result
    
    
    
    
    def get_convolution_product(self, t: float) -> np.ndarray: # this is the sum of all the convolution product of phi with itself
        return self.convolution_product_matrix(self.phi)(t)
    
    def get_convolution_product_matrix(self, t: float) -> np.ndarray: # this is the sum of all the convolution product of phi with itself
        return self.convolution_product_matrix(self.phi)(t)
    
    
    
    def iterate_convolution_product(self,function: callable) -> np.ndarray: # this is the sum of all the convolution product of phi with itself
        temp_function = function
        for index in range(self.convolution_threshold-1):
            temp_function = self.convolution_product(temp_function)
        return temp_function
    
    
    
    def iterate_convolution_product_matrix(self,function: callable) -> np.ndarray: # this is the sum of all the convolution product of phi with itself
        temp_function = function
        for index in range(self.convolution_threshold-1):
            temp_function = self.convolution_product_matrix(temp_function)
        return temp_function
    
    def get_iterate_convolution_product(self,function: callable, t: float) -> np.ndarray: # this is the sum of all the convolution product of phi with itself
        return self.iterate_convolution_product(function)(t)
    
    def  get_iterate_convolution_product_matrix(self,function: callable, t: float) -> np.ndarray: # this is the sum of all the convolution product of phi with itself
        return self.iterate_convolution_product_matrix(function)(t)
    
    
    
    
    # application of convolution product to phi & psi
    def get_psi_matrix(self, t: float) -> np.ndarray: # this is the sum of all the convolution product of phi with itself
        psi_matrix = np.zeros((self.dim, self.dim))
        for i in range(self.dim):
            for j in range(self.dim):
                psi_matrix[i,j] = self.convolution_product(self.phi, t, i, j)
        return psi_matrix
    
    
    
    
    
    def nu(self, t: float) -> float: # infinitesimal covariance matrix
        
        
        
        return
        
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
    def simulate(self, T: int, n: int):
        return 



    def fit(self, X: np.ndarray): # X is a numpy array of events
        return


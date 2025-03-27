import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import polars as pl
import pandas as pd
from scipy.integrate import quad

class Hawkes: # N dimensionnal hawkes process
    def __init__(self, phi: callable, psi: callable, mu: float, N: int, stepsize: int = 10):
        # phi returns a N dimensionnal vector
        # psi returns a N dimensionnal vector
        # mu is a N dimensionnal vector
        self.dim = N
        self.phi = phi 
        self.psi = psi
        self.convolution_threshold = 10 # number of time we convolve phi with itself
        self.mu = mu
        self.sigma = None
        self.nu = None
        self.mean_vector = None
        self.l1_norm_phi = None
        self.list_of_events = [[] for _ in range(N)] # matrix of time events, of size N x T
        self.stepsize = stepsize
        self.quadrature_points = np.linspace(0, 1, self.stepsize) # stepsize will change in the future
        self.quadrature_weights = np.ones(self.stepsize) / self.stepsize # weights also
        
        
        
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
        if self.mean_vector is None:
            mean_vector = np.zeros(self.dim)
            I = np.eye(self.dim)
            if self.l1_norm_phi is None:
                l1_matrix = self.get_l1_norm_phi()
                mean_vector = np.linalg.inv(I - l1_matrix) @ self.mu 
            self.mean_vector = mean_vector
        return self.mean_vector
    
    
    
    
    
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
    
    
    
    def convolve_functions(self, function1: callable, function2: callable) -> callable:
        return lambda t: quad(lambda tau: function1(t - tau) * function2(tau), 0, t)[0]
    
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
    
    def get_psi_function(self) -> callable: # this is the sum of all the convolution product of phi with itself
        self.psi_function = self.convolution_product_matrix(self.psi)
        return self.psi_function

    def get_psi(self, t: float) -> np.ndarray: # this is the sum of all the convolution product of phi with itself
        if self.psi_function is None:
            self.get_psi_function() 
        return self.psi_function(t)
    
    
    
    
    def get_sigma(self, t: float) -> np.ndarray: # matrix whose diagonal entries are the average intensity and the off-diagonal entries are the average intensity of the other dimensions
        mean_vector = self.get_average_intensity(t) # it is a vector, we need to make it a diagonal matrix
        mean_vector_matrix = np.diag(mean_vector)
        sigma = mean_vector_matrix
        return sigma
    
    def get_sigma_function(self) -> callable:
        return lambda t: self.get_sigma(t)
    
    
    # this is according to the formula (5) of Bacry et al
    def get_nu(self, t: float) -> float:
        if self.nu_function is None:
            sigma_function = self.get_sigma_function()
            psi_function = self.get_psi_function()
            value = self.convolve_functions(psi_function, lambda u: psi_function(u).T)(t) + sigma_function(t)@psi_function(t).T+psi_function(t)@sigma_function(t)+ sigma_function(t)
            return value
        return self.nu_function(t)

    def get_nu_function(self) -> callable:
        return lambda t: self.get_nu(t) # be careful
   
   
   # conditional laws g
    def get_g(self, t: float) -> float:
        if t <=0:
            return -np.eye(self.dim)
        return self.nu(t) * np.linalg.inv(self.get_sigma(t)) # there is a dirac term in 0
   
   # this function is solution of the wiener hopf system
    def get_g_function(self) -> callable:
        return lambda t: self.get_g(t)

   # this function is solution of the wiener hopf system
   
   
   
   
   
   
   # estimation functions
    
    def get_system(self) -> tuple[np.ndarray, np.ndarray]:
        K = self.stepsize
        D = self.dim
        
        system = np.zeros((D*K*K, D*K*K)) # we have D*K*K equations
        vector = np.zeros(D*K*K) # we have D*K*K unknowns
        
        # for each quadrature point and each dimension
        for n in range(K):
            for i in range(D):
                for j in range(D):
                    # Position dans le système linéaire
                    row = i*D*D + j*D + n
                    col = i*D*D + j*D + n
                    # diagonal term
                    system[row, col] = 1.0
                    # convolution terms
                    for l in range(D):
                        for k in range(K):
                            t_n = self.quadrature_points[n]
                            t_k = self.quadrature_points[k]
                            w_k = self.quadrature_weights[k]
                            system[row, l*D*D + j*D + k] = w_k * self.get_g(t_n - t_k)[i,l]
                    vector[row] = self.get_g(t_n)[i,j]
        return system, vector
    
   
   
   
   
   
    def verify_system(self) -> tuple[float, float]:
        # we check if the system is well conditioned
        system, vector = self.get_system()
        print(np.linalg.cond(system))
        
        # we check if invertible
        print(np.linalg.det(system) != 0)
        return np.linalg.cond(system), np.linalg.det(system)
    
    
    def get_estimator_phi(self, t: float) -> np.ndarray: # we estimate at sensor points
        system, vector = self.get_system()
        return np.linalg.inv(system) @ vector
    
    
    def reconstruct_phi(self) -> np.ndarray:
        vector = self.get_estimator_phi(0)
        K = self.stepsize
        D = self.dim
        phi_matrix = np.zeros((D, D,K))
        for i in range(D):
            for j in range(D):
                for k in range(K):
                    phi_matrix[i,j,k]= vector[i*D*D + j*D + k]
        return phi_matrix
    
    
    def get_estimator_nu(self, t: float) -> np.ndarray:
       return 
    
    
    def get_estimator_sigma(self, t: float) -> np.ndarray:
       return
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
    def simulate(self, T: int, n: int):
        return 



    def fit(self, X: np.ndarray): # X is a numpy array of events
        return


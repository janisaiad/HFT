import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import polars as pl
import pandas as pd


class Hawkes: # N dimensionnal hawkes process
    def __init__(self, phi: callable, psi: callable, mu: float, N: int):
        # phi returns a N dimensionnal vector
        # psi returns a N dimensionnal vector
        # mu is a N dimensionnal vector
        self.dim = N
        self.phi = phi
        self.psi = psi
        self.mu = mu
        self.list_of_events = [[] for _ in range(N)] # matrix of time events, of size N x T
        
        
        
    def add_event(self, event: float, dimension: int):
        self.list_of_events[dimension].append(event)
    
    
    def get_intensity(self, t: float) -> np.ndarray: # returns a vector of intensities
        return self.mu + sum(self.psi(t - event) for event in self.list_of_events)
    
    
    
    
    
    
    
    
    
    def simulate(self, T: int, n: int):
        pass



    def fit(self, X: np.ndarray): # X is a numpy array of events
        pass


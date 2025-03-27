import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import polars as pl
import pandas as pd


class Hawkes:
    def __init__(self, phi: callable, psi: callable, mu: float):
        self.phi = phi
        self.psi = psi
        self.mu = mu
        self.list_of_events = [] # list of time numpy to use after
        
    def add_event(self, event: float):
        self.list_of_events.append(event)
    
    
    def get_intensity(self, t: float):
        return self.mu + sum(self.psi(t - event) for event in self.list_of_events)
    
    
    
    def simulate(self, T: int, n: int):
        pass

    def fit(self, X: np.ndarray):
        pass


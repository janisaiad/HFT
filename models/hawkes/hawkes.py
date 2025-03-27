import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import polars as pl
import pandas as pd
from scipy.integrate import quad
import cupy as cp
from tqdm import tqdm



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
        self.quadrature_points = cp.linspace(0, 1, self.stepsize) # stepsize will change in the future
        self.quadrature_weights = cp.ones(self.stepsize) / self.stepsize # weights also
        self.results = None
        
        
    def add_event(self, event: float, dimension: int):
        self.list_of_events[dimension].append(event)
    
   
   
   
    
    # EVENTS
    def print_events(self):
        for i in range(self.dim):
            print(self.list_of_events[i])
        return self.list_of_events
    
    def get_events(self, t: float) -> cp.ndarray:
        return self.list_of_events[t]
    
    def verify_l1_norm_phi(self) -> bool:
        l1_matrix = cp.zeros((self.dim, self.dim))
        for i in range(self.dim):
            for j in range(self.dim):
                def integrand(t):
                    return cp.abs(self.phi(t)[i,j])
                l1_matrix[i,j], _ = quad(integrand, 0, cp.inf) # this is managed by scipy
        spectral_radius = cp.max(cp.abs(cp.linalg.eigvals(l1_matrix)))
        self.l1_norm_phi = l1_matrix
        return spectral_radius < 1
    
    
    
    # L1 NORM
    
    def print_l1_norm_phi(self):
        l1_matrix = cp.zeros((self.dim, self.dim))
        for i in range(self.dim):
            for j in range(self.dim):
                l1_matrix[i,j] = quad(lambda t: cp.abs(self.phi(t)[i,j]), 0, cp.inf)[0]
        print(l1_matrix)
        self.l1_norm_phi = l1_matrix
        return l1_matrix
    
    def get_l1_norm_phi(self):
        l1_matrix = cp.zeros((self.dim, self.dim))
        for i in range(self.dim):
            for j in range(self.dim):
                l1_matrix[i,j] = quad(lambda t: cp.abs(self.phi(t)[i,j]), 0, cp.inf)[0]
        self.l1_norm_phi = l1_matrix
        return l1_matrix
    
    
    # INTENSITIES
    def print_intensity(self, t: float):
        intensity = self.get_intensity(t)
        print(intensity)
        return intensity

    def get_intensity(self, t: float) -> cp.ndarray: # returns a vector of intensities
        intensity = cp.array(self.mu.copy())
        for dim in range(self.dim):
            for event in self.list_of_events[dim]:
                intensity += self.phi(t - event)[:, dim]  # phi returns a matrix, take column dim
        return intensity
    
    
    
    
    # here we compute the equation (4) of the pape Bacry et al
    def get_average_intensity(self) -> float:
        if self.mean_vector is None:
            mean_vector = cp.zeros(self.dim)
            I = cp.eye(self.dim)
            if self.l1_norm_phi is None:
                l1_matrix = self.get_l1_norm_phi()
                mean_vector = cp.linalg.inv(I - l1_matrix) @ cp.array(self.mu)
            self.mean_vector = mean_vector
        return self.mean_vector
    
    
    
    
    
    # part 2.2 of Bacry et al
    
    # convolution product utils, to be put in utils folder after
    def convolution_product(self, function: callable) -> callable: # this is the convolution product of
        return lambda t: quad(lambda tau: function(t - tau) * function(tau), 0, t)[0] # we return a function of t
    
    
    def convolution_product_matrix(self, function: callable) -> callable:
        
        def result(t)->cp.ndarray:
            psi_matrix = cp.zeros((self.dim, self.dim))
            for i in range(self.dim):
               for j in range(self.dim):
                   function_to_apply = lambda tau: function(tau)[i,j]
                   psi_matrix[i,j] = self.convolution_product(function_to_apply)(t)
            return psi_matrix
        
        return result
    
    
    
    def convolve_functions(self, function1: callable, function2: callable) -> callable:
        return lambda t: quad(lambda tau: function1(t - tau) * function2(tau), 0, t)[0]
    
    def get_convolution_product(self, t: float) -> cp.ndarray: # this is the sum of all the convolution product of phi with itself
        return self.convolution_product_matrix(self.phi)(t)
    
    def get_convolution_product_matrix(self, t: float) -> cp.ndarray: # this is the sum of all the convolution product of phi with itself
        return self.convolution_product_matrix(self.phi)(t)
    
    
    
    def iterate_convolution_product(self,function: callable) -> cp.ndarray: # this is the sum of all the convolution product of phi with itself
        temp_function = function
        for index in range(self.convolution_threshold-1):
            temp_function = self.convolution_product(temp_function)
        return temp_function
    
    
    
    def iterate_convolution_product_matrix(self,function: callable) -> cp.ndarray: # this is the sum of all the convolution product of phi with itself
        temp_function = function
        for index in range(self.convolution_threshold-1):
            temp_function = self.convolution_product_matrix(temp_function)
        return temp_function
    
    def get_iterate_convolution_product(self,function: callable, t: float) -> cp.ndarray: # this is the sum of all the convolution product of phi with itself
        return self.iterate_convolution_product(function)(t)
    
    def  get_iterate_convolution_product_matrix(self,function: callable, t: float) -> cp.ndarray: # this is the sum of all the convolution product of phi with itself
        return self.iterate_convolution_product_matrix(function)(t)
    
    
    
    
    
    
    # application of convolution product to phi & psi
    
    def get_psi_function(self) -> callable: # this is the sum of all the convolution product of phi with itself
        self.psi_function = self.convolution_product_matrix(self.psi)
        return self.psi_function

    def get_psi(self, t: float) -> cp.ndarray: # this is the sum of all the convolution product of phi with itself
        if self.psi_function is None:
            self.get_psi_function() 
        return self.psi_function(t)
    
    
    
    
    def get_sigma(self, t: float) -> cp.ndarray: # matrix whose diagonal entries are the average intensity and the off-diagonal entries are the average intensity of the other dimensions
        mean_vector = self.get_average_intensity(t) # it is a vector, we need to make it a diagonal matrix
        mean_vector_matrix = cp.diag(mean_vector)
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
            return -cp.eye(self.dim)
        return self.nu(t) * cp.linalg.inv(self.get_sigma(t)) # there is a dirac term in 0
   
   # this function is solution of the wiener hopf system
    def get_g_function(self) -> callable:
        return lambda t: self.get_g(t)

   # this function is solution of the wiener hopf system
   
   
   
   
   
   
   #data handling
   
    def get_parquet(self,path: str) -> pl.DataFrame:
        return pl.read_parquet(path)
    
    
    
    
    
    # we should compute some g integrals
    def get_g_from_parquet(self,df: pl.DataFrame,threshold: int = 1000) -> cp.ndarray: # we estimate g in the time grid 
        # pour chaque colonne on regarde les temps d'arrivée
        df = df.with_columns(
        pl.col("ts_event").cast(pl.Datetime).alias("timestamp")
)
        # df = df.slice(0, threshold)
        pdf = df.to_pandas()
        pdf["timestamp"] = pd.to_datetime(pdf["timestamp"])
        
        # Définir le temps de référence
        t0 = pdf["timestamp"].min()
        pdf["time_microseconds"] = (pdf["timestamp"] - t0).dt.total_seconds()*1000000
        # Define event types and create event_type column based on which column changed
        event_types = ["P_a", "P_b", "T_a", "T_b", "L_a", "L_b", "C_a", "C_b"]
        event_to_idx = {evt: i for i, evt in enumerate(event_types)}
        self.dim = len(event_types)
        
        # Create event_type column by checking which column had a change
        def determine_event_type(row):
            for evt in event_types:
                if pd.notna(row[evt]):
                    return event_to_idx[evt]
            return None
            
        pdf["event_type"] = pdf.apply(determine_event_type, axis=1)
        
        # Group events by type
        events_by_type = [pdf[pdf["event_type"] == i]["time_microseconds"].values for i in range(self.dim)]
        # Calculer les intensités de base (Lambda^i)
        total_duration = pdf["time_microseconds"].max() - pdf["time_microseconds"].min()
        lambda_i = cp.array([len(events) / total_duration for events in events_by_type])
        
        max_lag = 300 # Tmax
        n_bins = self.stepsize
        t_grid = self.quadrature_points
        
        
        g_estimates = cp.zeros((self.dim, self.dim, n_bins))
        g_integrals = cp.zeros((self.dim, self.dim, n_bins))  # ∫g(u)du
        ug_integrals = cp.zeros((self.dim, self.dim, n_bins))  # ∫ug(u)du
        
            
        # Pour chaque paire de types d'événements (i,j)
        for j in tqdm(range(self.dim), desc="Computing g_estimates"):
            j_events = events_by_type[j]
            
            for i in tqdm(range(self.dim), desc="Computing g_estimates"):
                i_events = events_by_type[i]
                
                if len(j_events) == 0 or len(i_events) == 0:
                    continue
                
                # Histogramme des délais entre événements j et i
                # Attention: pour n_bins points sur la grille, il y a n_bins+1 délimiteurs de bins
                counts = cp.zeros(n_bins)  # devrait être de taille n_bins
                total_j_events = 0
                
                for t_j in j_events:
                    # Trouver tous les événements i qui suivent l'événement j dans la fenêtre max_lag
                    deltas = i_events[i_events > t_j] - t_j
                    deltas = deltas[deltas <= max_lag]
                    
                    if len(deltas) > 0:
                        # Calculer l'histogramme des délais
                        # Les bins doivent être spécifiés comme un tableau de taille n_bins+1
                        bin_edges = cp.linspace(0, max_lag, n_bins + 1)  # n_bins+1 points pour n_bins intervalles
                        hist, _ = cp.histogram(cp.array(deltas), bins=bin_edges)
                        
                        # Maintenant hist et counts ont la même taille (n_bins)
                        counts += hist
                        total_j_events += 1
                
                if total_j_events > 0:
                    # Normaliser pour obtenir une estimation de E[dN^i_t|dN^j_0=1]
                    bin_width = t_grid[1] - t_grid[0]
                    intensity_conditional = counts / (total_j_events * bin_width)
                    
                    # on calcule g_ij(t) = E[dN^i_t|dN^j_0=1] - 1_{i=j}δ(t) - Λ^i
                    # Note: δ(t) est gérée implicitement car l'histogramme commence à t > 0
                    g_estimates[i, j, :] = intensity_conditional - lambda_i[i]
                    
                    #  les intégrales pour l'équation de Wiener-Hopf
                    for n in range(n_bins):
                        # sur chaque point de temps t_n
                        t_n = t_grid[n]
                        
                        #  ∫g(u)du pour chaque intervalle [t_n-t_{k+1}, t_n-t_k]
                        for k in range(n_bins):
                            if k < n:  #pour que t_n-t_k > 0
                                t_k = t_grid[k]
                                t_k_plus_1 = t_grid[k+1] if k+1 < n_bins else t_grid[k] + bin_width
                                
                                # indices pour l'intervalle [t_n-t_{k+1}, t_n-t_k]
                                start_idx = max(0, int((t_n - t_k_plus_1) / bin_width))
                                end_idx = min(n_bins-1, int((t_n - t_k) / bin_width))
                                
                                if start_idx <= end_idx:
                                    # ∫g(u)du sur l'intervalle
                                    integral_range = g_estimates[i, j, start_idx:end_idx+1]
                                    g_integrals[i, j, k] += cp.sum(integral_range) * bin_width
                                    
                                    # ∫ug(u)du sur l'intervalle
                                    u_values = t_grid[start_idx:end_idx+1]
                                    ug_integrals[i, j, k] += cp.sum(u_values * integral_range) * bin_width
        
    
        results = {
            "g_estimates": g_estimates,
            "g_integrals": g_integrals,
            "ug_integrals": ug_integrals,
            "t_grid": t_grid,
            "lambda_i": lambda_i
        }
        self.results = results
        return results
    
    
    def compute_wiener_hopf_linear_kernel(self, g_results=None, t_n_idx=0):
        g_results =  self.g_results if g_results is None else g_results
        
        
        g_estimates = g_results["g_estimates"]
        g_integrals = g_results["g_integrals"]
        ug_integrals = g_results["ug_integrals"]
        t_grid = g_results["t_grid"]
            
        # Dimensions du système
        K = self.stepsize  # Nombre de points de quadrature
        D = self.dim       # Dimension du processus Hawkes
        
        # Initialiser le système
        A = cp.zeros((D * D * K, D * D * K))
        b = cp.zeros(D * D * K)
        
        # Pour chaque point t_n et chaque paire (i,j)
        for n in range(K):
            t_n = t_grid[n]
            
            for i in range(D):
                for j in range(D):
                    # Indice dans le système linéaire pour la ligne correspondant à φ˜_ij(t_n)
                    row_idx = i * D * K + j * K + n
                    
                    # Termes de l'équation de Wiener-Hopf
                    
                    # 1. Terme φ˜_ij(t_n) (coefficient diagonal = 1)
                    col_idx_n = i * D * K + j * K + n
                    A[row_idx, col_idx_n] = 1.0
                    
                    # 2. Termes avec φ˜_il(t_k) pour tous l,k
                    for k in range(K):
                        t_k = t_grid[k]
                        
                        # Pour l'intégrale de g
                        for l in range(D):
                            # Indice de la colonne pour φ˜_il(t_k)
                            col_idx_k = i * D * K + l * K + k
                            
                            # Ajouter le coefficient de φ˜_il(t_k)
                            A[row_idx, col_idx_k] += g_integrals[l, j, k]
                            
                            # Si k+1 < K, ajouter les termes avec (φ˜(t_{k+1}) - φ˜(t_k))
                            if k+1 < K:
                                t_k_plus_1 = t_grid[k+1]
                                dt = t_k_plus_1 - t_k
                                
                                # Indice de la colonne pour φ˜_il(t_{k+1})
                                col_idx_k_plus_1 = i * D * K + l * K + (k+1)
                                
                                # Terme avec (t_n - t_k)/(t_{k+1} - t_k) * ∫g
                                coef_1 = (t_n - t_k) / dt
                                A[row_idx, col_idx_k_plus_1] += coef_1 * g_integrals[l, j, k]
                                A[row_idx, col_idx_k] -= coef_1 * g_integrals[l, j, k]
                                
                                # Terme avec -1/(t_{k+1} - t_k) * ∫ug
                                coef_2 = -1.0 / dt
                                A[row_idx, col_idx_k_plus_1] += coef_2 * ug_integrals[l, j, k]
                                A[row_idx, col_idx_k] -= coef_2 * ug_integrals[l, j, k]
                    
                    # Terme constant g˜_ij(t_n)
                    b[row_idx] = g_estimates[i, j, n]
        
        return A, b




    def solve_phi_from_wiener_hopf(self, g_results=None):
        """
        Résout le système linéaire issu de l'équation de Wiener-Hopf pour obtenir φ˜
        
        Args:
            g_results: Résultats de get_g_from_parquet. Si None, utilise self.results.
        
        Returns:
            Une matrice 3D de taille (D, D, K) contenant les valeurs de φ˜_ij(t_k)
        """
        # Construire le système
        A, b = self.compute_wiener_hopf_linear_kernel(g_results)
        
        # Vérifier le conditionnement du système
        cond_number = cp.linalg.cond(A)
        print(f"Conditionnement du système: {cond_number}")
        
        # Si le conditionnement est trop élevé, utiliser une méthode régularisée
        if cond_number > 1e8:
            print("Système mal conditionné. Utilisation de la régularisation de Tikhonov.")
            # Paramètre de régularisation
            alpha = 1e-6
            # Résolution avec régularisation
            x, residuals, rank, s = cp.linalg.lstsq(A, b, rcond=alpha)
        else:
            # Résolution standard
            x = cp.linalg.solve(A, b)
        
        # Reconstruction de φ˜
        K = self.stepsize
        D = self.dim
        phi_values = cp.zeros((D, D, K))
        
        for i in range(D):
            for j in range(D):
                for k in range(K):
                    idx = i * D * D + j * K + k
                    phi_values[i, j, k] = x[idx]
        
        # Créer une fonction φ˜ interpolée
        t_grid = g_results["t_grid"] if g_results else self.results["t_grid"]
        
        def phi_tilde(t):
            # Trouver l'indice le plus proche dans la grille
            if t < t_grid[0] or t > t_grid[-1]:
                return cp.zeros((D, D))
            
            # Interpolation linéaire
            idx = cp.searchsorted(t_grid, t)
            if idx == 0:
                return phi_values[:, :, 0]
            elif idx == len(t_grid):
                return phi_values[:, :, -1]
            else:
                t0, t1 = t_grid[idx-1], t_grid[idx]
                phi0, phi1 = phi_values[:, :, idx-1], phi_values[:, :, idx]
                alpha = (t - t0) / (t1 - t0)
                return (1 - alpha) * phi0 + alpha * phi1
        
        # Mettre à jour la fonction phi
        self.phi = phi_tilde
        
        return phi_values

        
        





























    
    def get_system(self) -> tuple[cp.ndarray, cp.ndarray]:
        K = self.stepsize
        D = self.dim
        
        system = cp.zeros((D*D*K, D*D*K)) # we have D*K*K equations
        vector = cp.zeros(D*D*K) # we have D*K*K unknowns
        
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
        print(cp.linalg.cond(system))
        
        # we check if invertible
        print(cp.linalg.det(system) != 0)
        return cp.linalg.cond(system), cp.linalg.det(system)
    
    
    def get_estimator_phi(self, t: float) -> cp.ndarray: # we estimate at sensor points
        system, vector = self.get_system()
        return cp.linalg.inv(system) @ vector
    
    
    def reconstruct_phi(self) -> cp.ndarray:
        vector = self.get_estimator_phi(0)
        K = self.stepsize
        D = self.dim
        phi_matrix = cp.zeros((D, D,K))
        for i in range(D):
            for j in range(D):
                for k in range(K):
                    phi_matrix[i,j,k]= vector[i*D*D + j*D + k]
        return phi_matrix
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
    def simulate(self, T: int, n: int):
        return 



    def fit(self, X: cp.ndarray): # X is a numpy array of events
        return


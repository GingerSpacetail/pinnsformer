"""
AI Agent Epidemiological Model using PINNs
==========================================

Models the spread of malicious behavior in multi-agent AI systems
using a modified SEIR epidemiological model.

Compartments:
- S(t): Susceptible agents (healthy, vulnerable)
- E(t): Exposed agents (attacked but latent)
- I(t): Infected agents (exhibiting malignant behavior)
- R(t): Removed agents (isolated/patched/immunized)

Author: V. Schastlivaia for AI Safety x Physics Grand Challenge
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim import LBFGS
from tqdm import tqdm

from model.ode_pinn import create_ode_pinn, init_ode_pinn_weights

class AIEpidemiologyModel:
    """
    AI Agent Epidemiological Model with configurable parameters
    """
    
    def __init__(self, 
                 beta=0.3,      # Attack transmission rate
                 sigma=0.1,     # Exposed to infected rate (1/incubation_period)
                 gamma=0.05,    # Recovery/removal rate
                 mu=0.01,       # Natural death rate
                 nu=0.02,       # Immunization rate
                 alpha=0.001,   # External attack rate
                 N=1000):       # Total population size
        
        self.beta = beta
        self.sigma = sigma
        self.gamma = gamma
        self.mu = mu
        self.nu = nu
        self.alpha = alpha
        self.N = N
        
        # Basic reproduction number
        self.R0 = beta / (gamma + mu)
        
    def seir_system(self, S, E, I, R, t):
        """
        SEIR system of ODEs for AI agent epidemiology
        
        Returns: dS/dt, dE/dt, dI/dt, dR/dt
        """
        # Force of infection (attack pressure)
        lambda_attack = self.beta * I / self.N + self.alpha
        
        # System of equations
        dSdt = -lambda_attack * S - self.nu * S + self.mu * self.N
        dEdt = lambda_attack * S - (self.sigma + self.mu + self.nu) * E
        dIdt = self.sigma * E - (self.gamma + self.mu) * I
        dRdt = self.gamma * I + self.nu * (S + E) - self.mu * R
        
        return dSdt, dEdt, dIdt, dRdt
    
    def analytical_solution(self, t, S0, E0, I0, R0):
        """
        Analytical solution for simple cases (linearized around equilibrium)
        Used for validation
        """
        # Simplified analytical solution for small perturbations
        # This is approximate and used mainly for testing
        
        # Equilibrium values
        S_eq = self.mu * self.N / (self.mu + self.nu + self.alpha)
        E_eq = self.alpha * S_eq / (self.sigma + self.mu + self.nu)
        I_eq = self.sigma * E_eq / (self.gamma + self.mu)
        R_eq = self.N - S_eq - E_eq - I_eq
        
        # For small perturbations, use exponential decay/growth
        # This is a simplified approximation
        if self.R0 < 1:
            decay_rate = (self.gamma + self.mu) * (1 - self.R0)
            S = S_eq + (S0 - S_eq) * np.exp(-decay_rate * t)
            E = E_eq + (E0 - E_eq) * np.exp(-decay_rate * t)
            I = I_eq + (I0 - I_eq) * np.exp(-decay_rate * t)
            R = R_eq + (R0 - R_eq) * np.exp(-decay_rate * t)
        else:
            growth_rate = (self.gamma + self.mu) * (self.R0 - 1)
            S = S_eq + (S0 - S_eq) * np.exp(-growth_rate * t)
            E = E_eq + (E0 - E_eq) * np.exp(growth_rate * t)
            I = I_eq + (I0 - I_eq) * np.exp(growth_rate * t)
            R = R_eq + (R0 - R_eq) * np.exp(growth_rate * t)
            
        return S, E, I, R

class SEIRPINNSolver:
    """
    Physics-Informed Neural Network solver for SEIR model
    """
    
    def __init__(self, model_params, device='cpu', network_type='ode_pinn'):
        self.model_params = model_params
        self.device = device
        self.network_type = network_type
        
        # Initialize neural network
        if network_type == 'ode_pinn':
            self.net = create_ode_pinn('seir', out_dim=4, hidden_dim=128, num_layer=4, enhanced=True).to(device)
            init_ode_pinn_weights(self.net)
        else:
            raise NotImplementedError(f"Network type '{network_type}' is not implemented")

        # Optimizer
        self.optimizer = LBFGS(self.net.parameters(), line_search_fn='strong_wolfe')
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
    
    def solve(self, t_domain, initial_conditions, num_epochs=1000, num_points=1000):
        """
        Solve SEIR system using PINNs
        
        Args:
            t_domain: [t_start, t_end]
            initial_conditions: [S0, E0, I0, R0]
            num_epochs: Training epochs
            num_points: Number of collocation points
        """
        t_start, t_end = t_domain
        S0, E0, I0, R0 = initial_conditions
        
        # Generate training data
        t_train = torch.linspace(t_start, t_end, num_points).reshape(-1, 1)
        t_train = t_train.to(self.device).requires_grad_(True)
        
        # Initial condition points
        t_ic = torch.tensor([[t_start]], dtype=torch.float32).to(self.device)
        
        loss_history = []
        
        print("Training PINN for SEIR model...")
        for _ in tqdm(range(num_epochs)):
            def closure():
                self.optimizer.zero_grad()
                
                # Physics loss (PDE residuals)
                if self.network_type == 'pinn':
                    pred = self.net(torch.zeros_like(t_train), t_train)  # (x=0, t=time) for ODE
                elif self.network_type == 'ode_pinn':
                    pred = self.net(t_train)  # Time-only input
                else:  # pinnsformer
                    pred = self.net(torch.zeros_like(t_train), t_train)
                
                S_pred, E_pred, I_pred, R_pred = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3], pred[:, 3:4]
                
                # Compute derivatives
                S_t = torch.autograd.grad(S_pred, t_train, grad_outputs=torch.ones_like(S_pred),
                                        retain_graph=True, create_graph=True)[0]
                E_t = torch.autograd.grad(E_pred, t_train, grad_outputs=torch.ones_like(E_pred),
                                        retain_graph=True, create_graph=True)[0]
                I_t = torch.autograd.grad(I_pred, t_train, grad_outputs=torch.ones_like(I_pred),
                                        retain_graph=True, create_graph=True)[0]
                R_t = torch.autograd.grad(R_pred, t_train, grad_outputs=torch.ones_like(R_pred),
                                        retain_graph=True, create_graph=True)[0]
                
                # SEIR system equations
                lambda_attack = self.model_params.beta * I_pred / self.model_params.N + self.model_params.alpha
                
                f_S = S_t + lambda_attack * S_pred + self.model_params.nu * S_pred - self.model_params.mu * self.model_params.N
                f_E = E_t - lambda_attack * S_pred + (self.model_params.sigma + self.model_params.mu + self.model_params.nu) * E_pred
                f_I = I_t - self.model_params.sigma * E_pred + (self.model_params.gamma + self.model_params.mu) * I_pred
                f_R = R_t - self.model_params.gamma * I_pred - self.model_params.nu * (S_pred + E_pred) + self.model_params.mu * R_pred
                
                loss_physics = torch.mean(f_S**2) + torch.mean(f_E**2) + torch.mean(f_I**2) + torch.mean(f_R**2)
                
                # Initial condition loss
                if self.network_type == 'pinn':
                    pred_ic = self.net(torch.zeros_like(t_ic), t_ic)
                elif self.network_type == 'ode_pinn':
                    pred_ic = self.net(t_ic)
                else:
                    pred_ic = self.net(torch.zeros_like(t_ic), t_ic)
                
                loss_ic = torch.mean((pred_ic[0, 0] - S0)**2) + \
                         torch.mean((pred_ic[0, 1] - E0)**2) + \
                         torch.mean((pred_ic[0, 2] - I0)**2) + \
                         torch.mean((pred_ic[0, 3] - R0)**2)
                
                # Conservation law (optional constraint)
                total_pop = S_pred + E_pred + I_pred + R_pred
                loss_conservation = torch.mean((total_pop - self.model_params.N)**2)
                
                # Total loss
                loss = loss_physics + 10 * loss_ic + 0.1 * loss_conservation
                
                loss.backward()
                loss_history.append([loss_physics.item(), loss_ic.item(), loss_conservation.item()])
                
                return loss
            
            self.optimizer.step(closure)
        
        self.loss_history = loss_history
        return self.net, loss_history
    
    def predict(self, t_test):
        """Predict SEIR values at test points"""
        t_test = torch.tensor(t_test, dtype=torch.float32).reshape(-1, 1).to(self.device)
        
        with torch.no_grad():
            if self.network_type == 'pinn':
                pred = self.net(torch.zeros_like(t_test), t_test)
            elif self.network_type == 'ode_pinn':
                pred = self.net(t_test)
            else:
                pred = self.net(torch.zeros_like(t_test), t_test)
            
            pred = pred.cpu().numpy()
            
        return pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]  # S, E, I, R

def analyze_scenarios():
    """
    Analyze different attack scenarios and mitigation strategies
    """
    scenarios = {
        'baseline': {'beta': 0.3, 'sigma': 0.1, 'gamma': 0.05, 'nu': 0.02, 'alpha': 0.001},
        'high_asr': {'beta': 0.8, 'sigma': 0.1, 'gamma': 0.05, 'nu': 0.02, 'alpha': 0.001},  # High attack success rate
        'fast_spread': {'beta': 0.3, 'sigma': 0.3, 'gamma': 0.05, 'nu': 0.02, 'alpha': 0.001},  # Fast incubation
        'strong_defense': {'beta': 0.3, 'sigma': 0.1, 'gamma': 0.2, 'nu': 0.1, 'alpha': 0.001},  # Fast detection + immunization
        'persistent_attack': {'beta': 0.3, 'sigma': 0.1, 'gamma': 0.05, 'nu': 0.02, 'alpha': 0.01},  # High external pressure
    }
    
    results = {}
    
    for scenario_name, params in scenarios.items():
        print(f"\nAnalyzing scenario: {scenario_name}")
        
        # Create model
        model = AIEpidemiologyModel(**params)
        print(f"R0 = {model.R0:.3f}")
        
        # Solve with ODE PINN
        solver = SEIRPINNSolver(model, device='cpu', network_type='ode_pinn')
        
        # Initial conditions: 1 infected agent in population of 1000
        initial_conditions = [999, 0, 1, 0]  # S0, E0, I0, R0
        t_domain = [0, 100]  # 100 time units
        
        net, loss_history = solver.solve(t_domain, initial_conditions, num_epochs=500, num_points=500)
        
        # Test predictions
        t_test = np.linspace(0, 100, 1000)
        S_pred, E_pred, I_pred, R_pred = solver.predict(t_test)
        
        results[scenario_name] = {
            'model': model,
            'time': t_test,
            'S': S_pred,
            'E': E_pred,
            'I': I_pred,
            'R': R_pred,
            'loss_history': loss_history
        }
    
    return results

if __name__ == "__main__":
    # Run scenario analysis
    results = analyze_scenarios()
    
    # Plot results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, (scenario, data) in enumerate(results.items()):
        ax = axes[i]
        
        ax.plot(data['time'], data['S'], 'b-', label='Susceptible', alpha=0.7)
        ax.plot(data['time'], data['E'], 'y-', label='Exposed', alpha=0.7)
        ax.plot(data['time'], data['I'], 'r-', label='Infected', alpha=0.7)
        ax.plot(data['time'], data['R'], 'g-', label='Removed', alpha=0.7)
        
        ax.set_title(f'{scenario}\nR₀ = {data["model"].R0:.3f}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Number of Agents')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Remove empty subplot
    if len(results) < len(axes):
        axes[-1].remove()
    
    plt.tight_layout()
    plt.savefig('ai_epidemiology_scenarios.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Summary statistics
    print("\n" + "="*50)
    print("SCENARIO ANALYSIS SUMMARY")
    print("="*50)
    
    for scenario, data in results.items():
        max_infected = np.max(data['I'])
        final_removed = data['R'][-1]
        time_to_peak = data['time'][np.argmax(data['I'])]
        
        print(f"\n{scenario.upper()}:")
        print(f"  R₀: {data['model'].R0:.3f}")
        print(f"  Peak infected: {max_infected:.1f} agents")
        print(f"  Time to peak: {time_to_peak:.1f}")
        print(f"  Final removed: {final_removed:.1f} agents")
        
        if data['model'].R0 > 1:
            print(f"  ⚠️  EPIDEMIC THRESHOLD EXCEEDED")
        else:
            print(f"  ✅ Below epidemic threshold")
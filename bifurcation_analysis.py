"""
Bifurcation Analysis for AI Agent Epidemiological Model
======================================================

Analyzes critical points, stability, and bifurcations in the AI agent
epidemiological system to identify intervention thresholds.

Key bifurcations:
1. Transcritical bifurcation at R_0 = 1 (epidemic threshold)
2. Saddle-node bifurcations for varying parameters
3. Hopf bifurcations for oscillatory behavior

Author: V. Schastlivaia for AI Safety x Physics Grand Challenge
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.linalg import eigvals
import seaborn as sns

class BifurcationAnalyzer:
    """
    Bifurcation analysis for AI agent epidemiological model
    """
    
    def __init__(self, base_params):
        """
        Initialize with base parameter set
        
        Args:
            base_params: Dictionary with beta, sigma, gamma, mu, nu, alpha, N
        """
        self.base_params = base_params.copy()
        
    def compute_equilibria(self, params):
        """
        Compute equilibrium points of the SEIR system
        
        Returns:
            dict: Disease-free and endemic equilibria
        """
        beta, sigma, gamma, mu, nu, alpha, N = (
            params['beta'], params['sigma'], params['gamma'],
            params['mu'], params['nu'], params['alpha'], params['N']
        )
        
        # disease-free equilibrium
        S_df = mu * N / (mu + nu + alpha)
        E_df = alpha * S_df / (sigma + mu + nu)
        I_df = 0
        R_df = N - S_df - E_df
        
        dfe = {'S': S_df, 'E': E_df, 'I': I_df, 'R': R_df}
        
        # endemic equilibrium (if R_0 > 1)
        R0 = beta / (gamma + mu)
        
        if R0 <= 1:
            endemic = None
        else:
            # solve nonlinear system for endemic equilibrium
            def endemic_equations(vars):
                S, E, I, R = vars
                lambda_attack = beta * I / N + alpha
                
                eq1 = -lambda_attack * S - nu * S + mu * N
                eq2 = lambda_attack * S - (sigma + mu + nu) * E
                eq3 = sigma * E - (gamma + mu) * I
                eq4 = S + E + I + R - N
                
                return [eq1, eq2, eq3, eq4]
            
            # initial guess close to disease-free equilibrium
            initial_guess = [S_df * 0.9, E_df * 1.1, 1, R_df]
            
            try:
                sol = fsolve(endemic_equations, initial_guess)
                if all(sol >= 0) and abs(sum(sol) - N) < 1e-6:
                    endemic = {'S': sol[0], 'E': sol[1], 'I': sol[2], 'R': sol[3]}
                else:
                    endemic = None
            except:
                endemic = None
        
        return {'disease_free': dfe, 'endemic': endemic, 'R0': R0}
    
    def jacobian_at_equilibrium(self, params, equilibrium):
        """
        Compute Jacobian matrix at equilibrium point
        
        Args:
            params: Model parameters
            equilibrium: Equilibrium point dict with S, E, I, R
            
        Returns:
            numpy.ndarray: 4x4 Jacobian matrix
        """
        beta, sigma, gamma, mu, nu, alpha, N = (
            params['beta'], params['sigma'], params['gamma'],
            params['mu'], params['nu'], params['alpha'], params['N']
        )
        
        S_eq, E_eq, I_eq, R_eq = equilibrium['S'], equilibrium['E'], equilibrium['I'], equilibrium['R']
        
        # partial derivatives of the SEIR system
        # F = [dS/dt, dE/dt, dI/dt, dR/dt]
        # J[i,j] = dF[i]/dx[j] where x = [S, E, I, R]
        
        J = np.zeros((4, 4))
        
        # dF/dS
        J[0, 0] = -beta * I_eq / N - alpha - nu  # d(dS/dt)/dS
        J[1, 0] = beta * I_eq / N + alpha         # d(dE/dt)/dS
        J[2, 0] = 0                               # d(dI/dt)/dS
        J[3, 0] = nu                              # d(dR/dt)/dS
        
        # dF/dE
        J[0, 1] = 0                               # d(dS/dt)/dE
        J[1, 1] = -(sigma + mu + nu)              # d(dE/dt)/dE
        J[2, 1] = sigma                           # d(dI/dt)/dE
        J[3, 1] = nu                              # d(dR/dt)/dE
        
        # dF/dI
        J[0, 2] = -beta * S_eq / N                # d(dS/dt)/dI
        J[1, 2] = beta * S_eq / N                 # d(dE/dt)/dI
        J[2, 2] = -(gamma + mu)                   # d(dI/dt)/dI
        J[3, 2] = gamma                           # d(dR/dt)/dI
        
        # dF/dR
        J[0, 3] = 0                               # d(dS/dt)/dR
        J[1, 3] = 0                               # d(dE/dt)/dR
        J[2, 3] = 0                               # d(dI/dt)/dR
        J[3, 3] = -mu                             # d(dR/dt)/dR
        
        return J
    
    def stability_analysis(self, params):
        """
        Analyze stability of equilibria
        
        Returns:
            dict: Stability information for each equilibrium
        """
        equilibria = self.compute_equilibria(params)
        results = {}
        
        for eq_name, eq_point in equilibria.items():
            if eq_name == 'R0':
                continue
            if eq_point is None:
                continue
                
            J = self.jacobian_at_equilibrium(params, eq_point)
            eigenvalues = eigvals(J)
            
            # determine stability
            real_parts = np.real(eigenvalues)
            is_stable = all(real_parts < 0)
            
            results[eq_name] = {
                'equilibrium': eq_point,
                'eigenvalues': eigenvalues,
                'stable': is_stable,
                'max_real_part': max(real_parts)
            }
        
        results['R0'] = equilibria['R0']
        return results
    
    def parameter_sweep_R0(self, param_name, param_range, num_points=100):
        """
        sweep parameter and track R_0 for bifurcation analysis
        
        Args:
            param_name: Name of parameter to vary
            param_range: [min_val, max_val] range
            num_points: Number of points to sample
            
        Returns:
            tuple: (parameter_values, R0_values, critical_points)
        """
        param_values = np.linspace(param_range[0], param_range[1], num_points)
        R0_values = []
        critical_points = []
        
        for param_val in param_values:
            params = self.base_params.copy()
            params[param_name] = param_val
            
            # compute R_0
            beta, gamma, mu = params['beta'], params['gamma'], params['mu']
            R0 = beta / (gamma + mu)
            R0_values.append(R0)
            
            # check for critical points (R_0 = 1)
            if len(R0_values) > 1:
                if (R0_values[-2] - 1) * (R0_values[-1] - 1) < 0:
                    # sign change detected - critical point
                    critical_points.append((param_val, 1.0))
        
        return np.array(param_values), np.array(R0_values), critical_points
    
    def bifurcation_diagram(self, param_name, param_range, num_points=100):
        """
        bifurcation diagram showing equilibria vs parameter
        
        Args:
            param_name: Parameter to vary
            param_range: [min, max] parameter range
            num_points: Number of parameter values
            
        Returns:
            dict: Bifurcation data
        """
        param_values = np.linspace(param_range[0], param_range[1], num_points)
        
        stable_branches = {'S': [], 'E': [], 'I': [], 'R': []}
        unstable_branches = {'S': [], 'E': [], 'I': [], 'R': []}
        param_stable = []
        param_unstable = []
        R0_values = []
        
        for param_val in param_values:
            params = self.base_params.copy()
            params[param_name] = param_val
            
            stability_info = self.stability_analysis(params)
            R0_values.append(stability_info['R0'])
            
            # Track stable and unstable equilibria
            for eq_name in ['disease_free', 'endemic']:
                if eq_name in stability_info and stability_info[eq_name] is not None:
                    eq_point = stability_info[eq_name]['equilibrium']
                    is_stable = stability_info[eq_name]['stable']
                    
                    if is_stable:
                        stable_branches['S'].append(eq_point['S'])
                        stable_branches['E'].append(eq_point['E'])
                        stable_branches['I'].append(eq_point['I'])
                        stable_branches['R'].append(eq_point['R'])
                        param_stable.append(param_val)
                    else:
                        unstable_branches['S'].append(eq_point['S'])
                        unstable_branches['E'].append(eq_point['E'])
                        unstable_branches['I'].append(eq_point['I'])
                        unstable_branches['R'].append(eq_point['R'])
                        param_unstable.append(param_val)
        
        return {
            'parameter': param_values,
            'R0': np.array(R0_values),
            'stable': {
                'parameter': np.array(param_stable),
                'branches': stable_branches
            },
            'unstable': {
                'parameter': np.array(param_unstable),
                'branches': unstable_branches
            }
        }
    
    def plot_bifurcation_diagram(self, param_name, param_range, compartment='I', figsize=(10, 6)):
        """
        plot bifurcation diagram for specified compartment
        
        Args:
            param_name: Parameter name for x-axis
            param_range: Parameter range
            compartment: Which compartment to plot ('S', 'E', 'I', 'R')
            figsize: Figure size
        """
        bifurc_data = self.bifurcation_diagram(param_name, param_range)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
        
        # plot bifurcation diagram
        if bifurc_data['stable']['parameter'].size > 0:
            ax1.plot(bifurc_data['stable']['parameter'], 
                    bifurc_data['stable']['branches'][compartment],
                    'b-', linewidth=2, label='Stable equilibrium')
        
        if bifurc_data['unstable']['parameter'].size > 0:
            ax1.plot(bifurc_data['unstable']['parameter'],
                    bifurc_data['unstable']['branches'][compartment],
                    'r--', linewidth=2, label='Unstable equilibrium')
        
        ax1.set_ylabel(f'{compartment} (Number of agents)')
        ax1.set_title(f'Bifurcation Diagram: {compartment} vs {param_name}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # plot R_0
        ax2.plot(bifurc_data['parameter'], bifurc_data['R0'], 'g-', linewidth=2)
        ax2.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='R₀ = 1 (Critical threshold)')
        ax2.set_xlabel(param_name)
        ax2.set_ylabel('R₀')
        ax2.set_title('Basic Reproduction Number')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def intervention_analysis(self, target_R0=0.8):
        """
        Analyze intervention strategies to achieve target R_0
        
        Args:
            target_R0: Target basic reproduction number
            
        Returns:
            dict: Required parameter changes for each intervention
        """
        interventions = {}
        current_R0 = self.base_params['beta'] / (self.base_params['gamma'] + self.base_params['mu'])
        
        if current_R0 <= target_R0:
            return {"message": f"Current R₀ = {current_R0:.3f} already below target {target_R0}"}
        
        # intervention 1: reduce attack transmission rate (\beta)
        required_beta = target_R0 * (self.base_params['gamma'] + self.base_params['mu'])
        interventions['reduce_transmission'] = {
            'parameter': 'beta',
            'current_value': self.base_params['beta'],
            'required_value': required_beta,
            'reduction_factor': required_beta / self.base_params['beta'],
            'description': 'Improve agent robustness, network segmentation'
        }
        
        # intervention 2: increase detection rate (\gamma)
        required_gamma = self.base_params['beta'] / target_R0 - self.base_params['mu']
        if required_gamma > 0:
            interventions['increase_detection'] = {
                'parameter': 'gamma',
                'current_value': self.base_params['gamma'],
                'required_value': required_gamma,
                'increase_factor': required_gamma / self.base_params['gamma'],
                'description': 'Improve monitoring, faster isolation'
            }
        
        # intervention 3: increase immunization rate (\nu)
        # this affects the equilibrium indirectly
        interventions['increase_immunization'] = {
            'parameter': 'nu',
            'current_value': self.base_params['nu'],
            'suggested_value': self.base_params['nu'] * 2,  # double immunization rate
            'description': 'Proactive patching, security updates'
        }
        
        return interventions

def demonstrate_bifurcation_analysis():
    """
    bifurcation analysis capabilities
    """
    # base parameters for AI agent system
    base_params = {
        'beta': 0.3,    # Attack transmission rate
        'sigma': 0.1,   # Exposed to infected rate
        'gamma': 0.05,  # Recovery rate
        'mu': 0.01,     # Death rate
        'nu': 0.02,     # Immunization rate
        'alpha': 0.001, # External attack rate
        'N': 1000       # Population size
    }
    
    analyzer = BifurcationAnalyzer(base_params)
    
    # analyze current stability
    print("STABILITY ANALYSIS")
    print("=" * 50)
    stability = analyzer.stability_analysis(base_params)
    
    for eq_name, info in stability.items():
        if eq_name == 'R0':
            print(f"Basic Reproduction Number R₀ = {info:.3f}")
            continue
        if info is None:
            continue
            
        print(f"\n{eq_name.upper()} EQUILIBRIUM:")
        print(f"  S = {info['equilibrium']['S']:.1f}")
        print(f"  E = {info['equilibrium']['E']:.1f}")
        print(f"  I = {info['equilibrium']['I']:.1f}")
        print(f"  R = {info['equilibrium']['R']:.1f}")
        print(f"  Stable: {info['stable']}")
        print(f"  Max eigenvalue real part: {info['max_real_part']:.4f}")
    
    # bifurcation analysis for transmission rate
    print(f"\nBIFURCATION ANALYSIS")
    print("=" * 50)
    
    fig1 = analyzer.plot_bifurcation_diagram('beta', [0.01, 0.5], compartment='I')
    plt.savefig('bifurcation_beta_infected.png', dpi=300, bbox_inches='tight')
    
    fig2 = analyzer.plot_bifurcation_diagram('gamma', [0.01, 0.2], compartment='I')
    plt.savefig('bifurcation_gamma_infected.png', dpi=300, bbox_inches='tight')
    
    # intervention analysis
    print(f"\nINTERVENTION ANALYSIS")
    print("=" * 50)
    
    interventions = analyzer.intervention_analysis(target_R0=0.8)
    
    for intervention_name, details in interventions.items():
        if 'message' in details:
            print(details['message'])
            continue
            
        print(f"\n{intervention_name.upper()}:")
        print(f"  Parameter: {details['parameter']}")
        print(f"  Current value: {details['current_value']:.4f}")
        
        if 'required_value' in details:
            print(f"  Required value: {details['required_value']:.4f}")
            if 'reduction_factor' in details:
                print(f"  Reduction needed: {(1-details['reduction_factor'])*100:.1f}%")
            elif 'increase_factor' in details:
                print(f"  Increase needed: {(details['increase_factor']-1)*100:.1f}%")
        elif 'suggested_value' in details:
            print(f"  Suggested value: {details['suggested_value']:.4f}")
            
        print(f"  Strategy: {details['description']}")
    
    plt.show()
    
    return analyzer

if __name__ == "__main__":
    analyzer = demonstrate_bifurcation_analysis()
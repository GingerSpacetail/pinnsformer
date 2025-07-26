"""
Phase Space Analysis for AI Agent Epidemiological Model
=======================================================

Phase space visualization and bifurcation analysis
for understanding dynamics of attack spread in AI agent populations.

Features:
- 2D and 3D phase portraits
- Nullcline analysis
- Vector field visualization
- Trajectory analysis with multiple initial conditions
- Bifurcation diagrams
- Stability analysis

Author: V. Schastlivaia for AI Safety x Physics Grand Challenge
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
from scipy.integrate import odeint
from scipy.optimize import fsolve
import seaborn as sns
from ai_epidemiology_model import AIEpidemiologyModel
from bifurcation_analysis import BifurcationAnalyzer

class PhaseSpaceAnalyzer:
    """
    phase space analysis for SEIR epidemiological model
    """
    
    def __init__(self, model_params):
        """
        initialize phase space analyzer
        
        Args:
            model_params: Dictionary of model parameters
        """
        self.params = model_params.copy()
        self.model = AIEpidemiologyModel(**model_params)
        self.bifurcation_analyzer = BifurcationAnalyzer(model_params)
        
    def seir_ode_system(self, state, t):
        """
        SEIR system for scipy.integrate.odeint
        
        Args:
            state: [S, E, I, R] current state
            t: time (not used in autonomous system)
            
        Returns:
            [dS/dt, dE/dt, dI/dt, dR/dt]
        """
        S, E, I, R = state
        dSdt, dEdt, dIdt, dRdt = self.model.seir_system(S, E, I, R, t)
        return [dSdt, dEdt, dIdt, dRdt]
    
    def compute_nullclines(self, var1='S', var2='I', resolution=100):
        """
        Compute nullclines for 2D phase plane analysis
        
        Args:
            var1: First variable ('S', 'E', 'I', 'R')
            var2: Second variable ('S', 'E', 'I', 'R')
            resolution: Grid resolution
            
        Returns:
            dictionary with nullcline data
        """
        # get equilibria to set reasonable bounds
        equilibria = self.bifurcation_analyzer.compute_equilibria(self.params)
        
        # Set bounds based on equilibria and population size
        bounds = {
            'S': (0, self.params['N']),
            'E': (0, self.params['N'] * 0.3),
            'I': (0, self.params['N'] * 0.3),
            'R': (0, self.params['N'])
        }
        
        v1_range = np.linspace(bounds[var1][0], bounds[var1][1], resolution)
        v2_range = np.linspace(bounds[var2][0], bounds[var2][1], resolution)
        V1, V2 = np.meshgrid(v1_range, v2_range)
        
        nullclines = {}
        
        # For each point, compute derivatives and find nullclines
        for i, (v1_val, v2_val) in enumerate(zip(V1.flatten(), V2.flatten())):
            # Set state vector (approximate other variables at equilibrium)
            if equilibria['endemic']:
                eq = equilibria['endemic']
            else:
                eq = equilibria['disease_free']
            
            state = [eq['S'], eq['E'], eq['I'], eq['R']]
            
            # Update the two variables we're analyzing
            var_indices = {'S': 0, 'E': 1, 'I': 2, 'R': 3}
            state[var_indices[var1]] = v1_val
            state[var_indices[var2]] = v2_val
            
            # Ensure conservation (approximately)
            total = sum(state)
            if total > self.params['N']:
                # Scale down proportionally
                scale = self.params['N'] / total
                state = [s * scale for s in state]
            
            derivatives = self.seir_ode_system(state, 0)
            
            # Store nullcline information
            if i == 0:
                nullclines['d' + var1 + '_dt'] = []
                nullclines['d' + var2 + '_dt'] = []
                nullclines[var1] = []
                nullclines[var2] = []
            
            nullclines['d' + var1 + '_dt'].append(derivatives[var_indices[var1]])
            nullclines['d' + var2 + '_dt'].append(derivatives[var_indices[var2]])
            nullclines[var1].append(v1_val)
            nullclines[var2].append(v2_val)
        
        # Convert to arrays and reshape
        for key in nullclines:
            nullclines[key] = np.array(nullclines[key]).reshape(V1.shape)
        
        nullclines['V1'] = V1
        nullclines['V2'] = V2
        nullclines['var1'] = var1
        nullclines['var2'] = var2
        
        return nullclines
    
    def plot_2d_phase_portrait(self, var1='S', var2='I', n_trajectories=10, t_max=200, figsize=(10, 8)):
        """
        Create 2D phase portrait with trajectories and nullclines
        
        Args:
            var1: First variable for phase plane
            var2: Second variable for phase plane
            n_trajectories: Number of trajectories to plot
            t_max: Maximum integration time
            figsize: Figure size
            
        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Compute nullclines
        nullclines = self.compute_nullclines(var1, var2)
        V1, V2 = nullclines['V1'], nullclines['V2']
        dV1_dt = nullclines['d' + var1 + '_dt']
        dV2_dt = nullclines['d' + var2 + '_dt']
        
        # Plot nullclines
        ax.contour(V1, V2, dV1_dt, levels=[0], colors='red', linewidths=2, alpha=0.7)
        ax.contour(V1, V2, dV2_dt, levels=[0], colors='blue', linewidths=2, alpha=0.7)
        
        # Plot vector field
        skip = 8  # Skip points for cleaner vector field
        ax.quiver(V1[::skip, ::skip], V2[::skip, ::skip], 
                 dV1_dt[::skip, ::skip], dV2_dt[::skip, ::skip],
                 alpha=0.5, scale=None, angles='xy', width=0.003)
        
        # Plot equilibrium points
        equilibria = self.bifurcation_analyzer.compute_equilibria(self.params)
        var_idx = {'S': 0, 'E': 1, 'I': 2, 'R': 3}
        
        # Disease-free equilibrium
        df_eq = equilibria['disease_free']
        stability_df = self.bifurcation_analyzer.stability_analysis(self.params)['disease_free']
        color_df = 'green' if stability_df['stable'] else 'red'
        marker_df = 'o' if stability_df['stable'] else 's'
        
        ax.plot(df_eq[var1], df_eq[var2], marker_df, markersize=12, 
               color=color_df, markeredgecolor='black', markeredgewidth=2,
               label=f'Disease-free ({"stable" if stability_df["stable"] else "unstable"})')
        
        # Endemic equilibrium
        if equilibria['endemic']:
            en_eq = equilibria['endemic']
            stability_en = self.bifurcation_analyzer.stability_analysis(self.params)['endemic']
            color_en = 'green' if stability_en['stable'] else 'red'
            marker_en = 'o' if stability_en['stable'] else 's'
            
            ax.plot(en_eq[var1], en_eq[var2], marker_en, markersize=12,
                   color=color_en, markeredgecolor='black', markeredgewidth=2,
                   label=f'Endemic ({"stable" if stability_en["stable"] else "unstable"})')
        
        # Plot trajectories with different initial conditions
        colors = plt.cm.viridis(np.linspace(0, 1, n_trajectories))
        
        for i in range(n_trajectories):
            # Generate random initial condition
            S0 = np.random.uniform(0.1 * self.params['N'], 0.9 * self.params['N'])
            E0 = np.random.uniform(0, 0.1 * self.params['N'])
            I0 = np.random.uniform(1, 0.1 * self.params['N'])
            R0 = max(0, self.params['N'] - S0 - E0 - I0)
            
            # Integrate trajectory
            t = np.linspace(0, t_max, 1000)
            trajectory = odeint(self.seir_ode_system, [S0, E0, I0, R0], t)
            
            # Extract variables for plotting
            traj_var1 = trajectory[:, var_idx[var1]]
            traj_var2 = trajectory[:, var_idx[var2]]
            
            # Plot trajectory
            ax.plot(traj_var1, traj_var2, color=colors[i], alpha=0.7, linewidth=1.5)
            ax.plot(traj_var1[0], traj_var2[0], 'o', color=colors[i], markersize=6, alpha=0.8)
            ax.plot(traj_var1[-1], traj_var2[-1], 's', color=colors[i], markersize=4, alpha=0.8)
        
        ax.set_xlabel(f'{var1} (Number of agents)')
        ax.set_ylabel(f'{var2} (Number of agents)')
        ax.set_title(f'Phase Portrait: {var2} vs {var1} (R₀ = {self.model.R0:.3f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_3d_phase_portrait(self, n_trajectories=8, t_max=200, figsize=(12, 10)):
        """
        Create 3D phase portrait in S-E-I space
        
        Args:
            n_trajectories: Number of trajectories to plot
            t_max: Maximum integration time
            figsize: Figure size
            
        Returns:
            Figure object
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot equilibrium points
        equilibria = self.bifurcation_analyzer.compute_equilibria(self.params)
        stability = self.bifurcation_analyzer.stability_analysis(self.params)
        
        # Disease-free equilibrium
        df_eq = equilibria['disease_free']
        color_df = 'green' if stability['disease_free']['stable'] else 'red'
        ax.scatter(df_eq['S'], df_eq['E'], df_eq['I'], 
                  s=200, c=color_df, marker='o', edgecolors='black', linewidth=2,
                  label=f'Disease-free ({"stable" if stability["disease_free"]["stable"] else "unstable"})')
        
        # Endemic equilibrium
        if equilibria['endemic']:
            en_eq = equilibria['endemic']
            color_en = 'green' if stability['endemic']['stable'] else 'red'
            ax.scatter(en_eq['S'], en_eq['E'], en_eq['I'],
                      s=200, c=color_en, marker='s', edgecolors='black', linewidth=2,
                      label=f'Endemic ({"stable" if stability["endemic"]["stable"] else "unstable"})')
        
        # Plot trajectories
        colors = plt.cm.plasma(np.linspace(0, 1, n_trajectories))
        
        for i in range(n_trajectories):
            # Generate initial condition
            S0 = np.random.uniform(0.2 * self.params['N'], 0.8 * self.params['N'])
            E0 = np.random.uniform(0, 0.15 * self.params['N'])
            I0 = np.random.uniform(1, 0.15 * self.params['N'])
            R0 = max(0, self.params['N'] - S0 - E0 - I0)
            
            # Integrate trajectory
            t = np.linspace(0, t_max, 1000)
            trajectory = odeint(self.seir_ode_system, [S0, E0, I0, R0], t)
            
            S_traj, E_traj, I_traj = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]
            
            # Plot trajectory
            ax.plot(S_traj, E_traj, I_traj, color=colors[i], alpha=0.8, linewidth=2)
            ax.scatter(S_traj[0], E_traj[0], I_traj[0], 
                      c=colors[i], s=80, marker='o', alpha=0.9, edgecolors='black')
            ax.scatter(S_traj[-1], E_traj[-1], I_traj[-1], 
                      c=colors[i], s=40, marker='s', alpha=0.9, edgecolors='black')
        
        ax.set_xlabel('S (Susceptible agents)')
        ax.set_ylabel('E (Exposed agents)')
        ax.set_zlabel('I (Infected agents)')
        ax.set_title(f'3D Phase Portrait: S-E-I Space (R₀ = {self.model.R0:.3f})')
        ax.legend()
        
        return fig
    
    def plot_bifurcation_with_phase_space(self, param_name='beta', param_range=None, figsize=(15, 12)):
        """
        Combined bifurcation diagram and phase space analysis
        
        Args:
            param_name: Parameter to vary
            param_range: Parameter range [min, max]
            figsize: Figure size
            
        Returns:
            Figure object
        """
        if param_range is None:
            if param_name == 'beta':
                param_range = [0.01, 0.8]
            elif param_name == 'gamma':
                param_range = [0.01, 0.3]
            elif param_name == 'nu':
                param_range = [0.001, 0.2]
            else:
                param_range = [0.01, 0.5]
        
        fig = plt.figure(figsize=figsize)
        
        # Bifurcation diagram (top)
        ax1 = plt.subplot(2, 2, (1, 2))
        bifurc_data = self.bifurcation_analyzer.bifurcation_diagram(param_name, param_range)
        
        # Plot stable and unstable branches
        if bifurc_data['stable']['parameter'].size > 0:
            ax1.plot(bifurc_data['stable']['parameter'], 
                    bifurc_data['stable']['branches']['I'],
                    'b-', linewidth=3, label='Stable equilibrium', alpha=0.8)
        
        if bifurc_data['unstable']['parameter'].size > 0:
            ax1.plot(bifurc_data['unstable']['parameter'],
                    bifurc_data['unstable']['branches']['I'],
                    'r--', linewidth=3, label='Unstable equilibrium', alpha=0.8)
        
        # Mark current parameter value
        current_val = self.params[param_name]
        ax1.axvline(x=current_val, color='green', linestyle=':', linewidth=2, 
                   alpha=0.7, label=f'Current {param_name} = {current_val:.3f}')
        
        ax1.set_xlabel(f'{param_name}')
        ax1.set_ylabel('I (Infected agents)')
        ax1.set_title(f'Bifurcation Diagram: Infected vs {param_name}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # R₀ plot
        ax2 = plt.subplot(2, 2, 3)
        ax2.plot(bifurc_data['parameter'], bifurc_data['R0'], 'g-', linewidth=2)
        ax2.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='R₀ = 1')
        ax2.axvline(x=current_val, color='green', linestyle=':', linewidth=2, alpha=0.7)
        ax2.set_xlabel(f'{param_name}')
        ax2.set_ylabel('R₀')
        ax2.set_title('Basic Reproduction Number')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Phase portrait for current parameters
        ax3 = plt.subplot(2, 2, 4)
        
        # Quick phase portrait
        nullclines = self.compute_nullclines('S', 'I', resolution=50)
        V1, V2 = nullclines['V1'], nullclines['V2']
        dV1_dt = nullclines['dS_dt']
        dV2_dt = nullclines['dI_dt']
        
        # Plot nullclines and vector field
        ax3.contour(V1, V2, dV1_dt, levels=[0], colors='red', linewidths=2, alpha=0.7)
        ax3.contour(V1, V2, dV2_dt, levels=[0], colors='blue', linewidths=2, alpha=0.7)
        
        skip = 4
        ax3.quiver(V1[::skip, ::skip], V2[::skip, ::skip], 
                  dV1_dt[::skip, ::skip], dV2_dt[::skip, ::skip],
                  alpha=0.5, scale=None, angles='xy', width=0.004)
        
        # Plot equilibria
        equilibria = self.bifurcation_analyzer.compute_equilibria(self.params)
        stability = self.bifurcation_analyzer.stability_analysis(self.params)
        
        df_eq = equilibria['disease_free']
        color_df = 'green' if stability['disease_free']['stable'] else 'red'
        ax3.plot(df_eq['S'], df_eq['I'], 'o', markersize=12, color=color_df, 
                markeredgecolor='black', markeredgewidth=2)
        
        if equilibria['endemic']:
            en_eq = equilibria['endemic']
            color_en = 'green' if stability['endemic']['stable'] else 'red'
            ax3.plot(en_eq['S'], en_eq['I'], 's', markersize=12, color=color_en,
                    markeredgecolor='black', markeredgewidth=2)
        
        ax3.set_xlabel('S (Susceptible)')
        ax3.set_ylabel('I (Infected)')
        ax3.set_title(f'Phase Portrait (R₀ = {self.model.R0:.3f})')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def intervention_phase_analysis(self, intervention_time=50, intervention_params=None, 
                                  t_max=200, figsize=(15, 10)):
        """
        Analyze phase space changes due to interventions
        
        Args:
            intervention_time: Time when intervention is applied
            intervention_params: New parameters after intervention
            t_max: Maximum simulation time
            figsize: Figure size
            
        Returns:
            Figure object
        """
        if intervention_params is None:
            # Default intervention: reduce beta by 70%, increase gamma by 200%
            intervention_params = self.params.copy()
            intervention_params['beta'] *= 0.3
            intervention_params['gamma'] *= 3.0
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Before intervention - phase portrait
        ax1 = axes[0, 0]
        
        # Quick phase portrait for pre-intervention
        nullclines_pre = self.compute_nullclines('S', 'I', resolution=40)
        V1, V2 = nullclines_pre['V1'], nullclines_pre['V2']
        dV1_dt = nullclines_pre['dS_dt']
        dV2_dt = nullclines_pre['dI_dt']
        
        ax1.contour(V1, V2, dV1_dt, levels=[0], colors='red', linewidths=2, alpha=0.7)
        ax1.contour(V1, V2, dV2_dt, levels=[0], colors='blue', linewidths=2, alpha=0.7)
        
        skip = 4
        ax1.quiver(V1[::skip, ::skip], V2[::skip, ::skip], 
                  dV1_dt[::skip, ::skip], dV2_dt[::skip, ::skip],
                  alpha=0.5, scale=None, angles='xy', width=0.004)
        
        ax1.set_xlabel('S (Susceptible)')
        ax1.set_ylabel('I (Infected)')
        ax1.set_title(f'Before Intervention (R₀ = {self.model.R0:.3f})')
        ax1.grid(True, alpha=0.3)
        
        # After intervention - phase portrait
        ax2 = axes[0, 1]
        
        model_post = AIEpidemiologyModel(**intervention_params)
        analyzer_post = PhaseSpaceAnalyzer(intervention_params)
        
        nullclines_post = analyzer_post.compute_nullclines('S', 'I', resolution=40)
        V1_post, V2_post = nullclines_post['V1'], nullclines_post['V2']
        dV1_dt_post = nullclines_post['dS_dt']
        dV2_dt_post = nullclines_post['dI_dt']
        
        ax2.contour(V1_post, V2_post, dV1_dt_post, levels=[0], colors='red', linewidths=2, alpha=0.7)
        ax2.contour(V1_post, V2_post, dV2_dt_post, levels=[0], colors='blue', linewidths=2, alpha=0.7)
        
        ax2.quiver(V1_post[::skip, ::skip], V2_post[::skip, ::skip], 
                  dV1_dt_post[::skip, ::skip], dV2_dt_post[::skip, ::skip],
                  alpha=0.5, scale=None, angles='xy', width=0.004)
        
        ax2.set_xlabel('S (Susceptible)')
        ax2.set_ylabel('I (Infected)')
        ax2.set_title(f'After Intervention (R₀ = {model_post.R0:.3f})')
        ax2.grid(True, alpha=0.3)
        
        # Time series comparison
        ax3 = axes[1, 0]
        
        # Simulate trajectory with intervention
        t_pre = np.linspace(0, intervention_time, 200)
        t_post = np.linspace(intervention_time, t_max, 300)
        
        # Initial condition
        initial_state = [950, 20, 25, 5]
        
        # Pre-intervention
        traj_pre = odeint(self.seir_ode_system, initial_state, t_pre)
        
        # Post-intervention (start from end of pre-intervention)
        state_at_intervention = traj_pre[-1]
        traj_post = odeint(analyzer_post.seir_ode_system, state_at_intervention, t_post)
        
        # Plot time series
        ax3.plot(t_pre, traj_pre[:, 0], 'b-', linewidth=2, label='S (pre)', alpha=0.8)
        ax3.plot(t_pre, traj_pre[:, 2], 'r-', linewidth=2, label='I (pre)', alpha=0.8)
        ax3.plot(t_post, traj_post[:, 0], 'b--', linewidth=2, label='S (post)', alpha=0.8)
        ax3.plot(t_post, traj_post[:, 2], 'r--', linewidth=2, label='I (post)', alpha=0.8)
        
        ax3.axvline(x=intervention_time, color='green', linestyle=':', linewidth=3, 
                   alpha=0.7, label='Intervention')
        
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Number of agents')
        ax3.set_title('Time Series with Intervention')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Combined phase trajectory
        ax4 = axes[1, 1]
        
        # Plot phase trajectory showing intervention effect
        ax4.plot(traj_pre[:, 0], traj_pre[:, 2], 'b-', linewidth=3, 
                label='Pre-intervention', alpha=0.8)
        ax4.plot(traj_post[:, 0], traj_post[:, 2], 'r-', linewidth=3, 
                label='Post-intervention', alpha=0.8)
        
        # Mark intervention point
        ax4.plot(state_at_intervention[0], state_at_intervention[2], 'go', 
                markersize=12, label='Intervention point')
        
        ax4.set_xlabel('S (Susceptible)')
        ax4.set_ylabel('I (Infected)')
        ax4.set_title('Phase Trajectory: Intervention Effect')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

def demonstrate_phase_space_analysis():
    """
    Comprehensive demonstration of phase space analysis capabilities
    """
    print("AI EPIDEMIOLOGY PHASE SPACE ANALYSIS")
    print("=" * 50)
    
    # Base parameters
    params = {
        'beta': 0.3,    # Attack transmission rate
        'sigma': 0.1,   # Exposed to infected rate
        'gamma': 0.05,  # Recovery rate
        'mu': 0.01,     # Death rate
        'nu': 0.02,     # Immunization rate
        'alpha': 0.001, # External attack rate
        'N': 1000       # Population size
    }
    
    analyzer = PhaseSpaceAnalyzer(params)
    
    print(f"Model parameters: R₀ = {analyzer.model.R0:.3f}")
    print(f"Epidemic threshold: {'EXCEEDED' if analyzer.model.R0 > 1 else 'not exceeded'}")
    
    # 1. 2D Phase portraits
    print("\\n1. Creating 2D phase portraits...")
    
    fig1 = analyzer.plot_2d_phase_portrait('S', 'I', n_trajectories=8)
    fig1.savefig('phase_portrait_S_I.png', dpi=300, bbox_inches='tight')
    
    fig2 = analyzer.plot_2d_phase_portrait('E', 'I', n_trajectories=8)
    fig2.savefig('phase_portrait_E_I.png', dpi=300, bbox_inches='tight')
    
    # 2. 3D Phase portrait
    print("2. Creating 3D phase portrait...")
    
    fig3 = analyzer.plot_3d_phase_portrait(n_trajectories=6)
    fig3.savefig('phase_portrait_3D.png', dpi=300, bbox_inches='tight')
    
    # 3. Bifurcation analysis with phase space
    print("3. Creating bifurcation analysis with phase space...")
    
    fig4 = analyzer.plot_bifurcation_with_phase_space('beta', [0.01, 0.6])
    fig4.savefig('bifurcation_phase_analysis.png', dpi=300, bbox_inches='tight')
    
    # 4. Intervention 
    print("4. Creating intervention phase analysis...")
    
    intervention_params = params.copy()
    intervention_params['beta'] *= 0.25  # 75% reduction in attack rate
    intervention_params['gamma'] *= 4.0  # 4x faster detection
    
    fig5 = analyzer.intervention_phase_analysis(
        intervention_time=40, 
        intervention_params=intervention_params
    )
    fig5.savefig('intervention_phase_analysis.png', dpi=300, bbox_inches='tight')
    
    print("\\n" + "=" * 50)
    print("PHASE SPACE ANALYSIS COMPLETE")
    print("=" * 50)
    print("Generated visualizations:")
    print("  - phase_portrait_S_I.png: S vs I phase plane")
    print("  - phase_portrait_E_I.png: E vs I phase plane")
    print("  - phase_portrait_3D.png: 3D S-E-I phase space")
    print("  - bifurcation_phase_analysis.png: Bifurcation + phase analysis")
    print("  - intervention_phase_analysis.png: Intervention effects")
    
    plt.close('all')  # Clean up
    
    return analyzer

if __name__ == "__main__":
    analyzer = demonstrate_phase_space_analysis()
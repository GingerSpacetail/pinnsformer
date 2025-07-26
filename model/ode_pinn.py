# ODE-specific PINN implementation for time-only systems
# It's for for epidemiological model and other time-dependent ODEs

import torch
import torch.nn as nn

class ODEPINNs(nn.Module):
    """
    Physics-Informed Neural Network for ODE systems with time-only input
    
    Designed for systems like:
    dy/dt = f(y, t)
    
    Where y can be a vector (e.g., [S, E, I, R] for SEIR model)
    """
    def __init__(self, out_dim, hidden_dim=128, num_layer=4):
        super(ODEPINNs, self).__init__()
        
        layers = []
        for i in range(num_layer-1):
            if i == 0:
                layers.append(nn.Linear(in_features=1, out_features=hidden_dim))  # time input only
                layers.append(nn.Tanh())
            else:
                layers.append(nn.Linear(in_features=hidden_dim, out_features=hidden_dim))
                layers.append(nn.Tanh())
        
        layers.append(nn.Linear(in_features=hidden_dim, out_features=out_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, t):
        """
        Forward pass with time input only
        
        Args:
            t: Time tensor of shape (batch_size, 1)
            
        Returns:
            y: Output tensor of shape (batch_size, out_dim)
        """
        return self.network(t)

class SinusoidalEmbedding(nn.Module):
    """
    Sinusoidal positional embedding for time inputs
    Helps PINN learn periodic and multi-scale temporal patterns
    """
    def __init__(self, embed_dim=64):
        super(SinusoidalEmbedding, self).__init__()
        self.embed_dim = embed_dim
        
    def forward(self, t):
        """
        Args:
            t: Time tensor of shape (batch_size, 1)
        Returns:
            embedded: Embedded time of shape (batch_size, embed_dim)
        """
        batch_size = t.shape[0]
        
        # create frequency multipliers
        freqs = torch.exp(torch.arange(0, self.embed_dim, 2, dtype=torch.float32, device=t.device) * 
                         (-torch.log(torch.tensor(10000.0)) / self.embed_dim))
        
        # compute sinusoidal embeddings
        args = t * freqs.unsqueeze(0)  # (batch_size, embed_dim//2)
        
        # interleave sin and cos
        embedding = torch.zeros(batch_size, self.embed_dim, device=t.device)
        embedding[:, 0::2] = torch.sin(args)
        embedding[:, 1::2] = torch.cos(args)
        
        return embedding

class EnhancedODEPINNs(nn.Module):
    """
    Enhanced ODE PINN with sinusoidal time embedding and residual connections
    """
    def __init__(self, out_dim, hidden_dim=128, num_layer=4, use_time_embedding=True, embed_dim=64):
        super(EnhancedODEPINNs, self).__init__()
        
        self.use_time_embedding = use_time_embedding
        
        if use_time_embedding:
            self.time_embedding = SinusoidalEmbedding(embed_dim)
            input_dim = embed_dim + 1  # embedded time + raw time
        else:
            input_dim = 1
        
        # build network with residual connections
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.Tanh()
        
        self.hidden_layers = nn.ModuleList()
        for i in range(num_layer - 2):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        self.output_layer = nn.Linear(hidden_dim, out_dim)
        
        # skip connection weights
        self.skip_connections = nn.ModuleList()
        for i in range(num_layer - 2):
            self.skip_connections.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
    
    def forward(self, t):
        """
        Forward pass with enhanced time processing
        
        Args:
            t: Time tensor of shape (batch_size, 1)
            
        Returns:
            y: Output tensor of shape (batch_size, out_dim)
        """
        if self.use_time_embedding:
            t_embed = self.time_embedding(t)
            x = torch.cat([t, t_embed], dim=-1)
        else:
            x = t
        
        # Input layer
        x = self.activation(self.input_layer(x))
        
        # Hidden layers with residual connections
        for i, (hidden_layer, skip_layer) in enumerate(zip(self.hidden_layers, self.skip_connections)):
            residual = x
            x = self.activation(hidden_layer(x))
            x = x + skip_layer(residual)  # Residual connection
        
        # Output layer
        x = self.output_layer(x)
        
        return x

class AdaptiveODEPINNs(nn.Module):
    """
    Adaptive ODE PINN that can dynamically adjust its behavior based on the solution characteristics
    """
    def __init__(self, out_dim, hidden_dim=128, num_layer=4):
        super(AdaptiveODEPINNs, self).__init__()
        
        # main network
        self.main_net = EnhancedODEPINNs(out_dim, hidden_dim, num_layer, use_time_embedding=True)
        
        # adaptive gating network (learns to weight different components)
        self.gate_net = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(hidden_dim // 4, out_dim),
            nn.Sigmoid()
        )
        
        # base solution network (for simple polynomial baseline)
        self.base_net = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, out_dim)
        )
    
    def forward(self, t):
        """
        Adaptive forward pass
        
        Args:
            t: Time tensor of shape (batch_size, 1)
            
        Returns:
            y: Adaptively weighted output
        """
        main_output = self.main_net(t)
        base_output = self.base_net(t)
        gates = self.gate_net(t)
        
        # adaptive combination
        output = gates * main_output + (1 - gates) * base_output
        
        return output

# factory function to create appropriate PINN for different problems
def create_ode_pinn(problem_type='seir', out_dim=4, hidden_dim=128, num_layer=4, enhanced=True):
    """
    Factory function to create ODE PINN based on problem type
    
    Args:
        problem_type: Type of ODE problem ('seir', 'sir', 'lotka_volterra', 'custom')
        out_dim: Output dimension
        hidden_dim: Hidden layer dimension
        num_layer: Number of layers
        enhanced: Whether to use enhanced version with time embedding
        
    Returns:
        PINN model suitable for the problem
    """
    if problem_type in ['seir', 'sir']:
        # epidemiological models benefit from enhanced features
        if enhanced:
            return EnhancedODEPINNs(out_dim, hidden_dim, num_layer, use_time_embedding=True, embed_dim=64)
        else:
            return ODEPINNs(out_dim, hidden_dim, num_layer)
    
    elif problem_type == 'lotka_volterra':
        # predator-prey models have oscillatory behavior, use adaptive
        return AdaptiveODEPINNs(out_dim, hidden_dim, num_layer)
    
    elif problem_type == 'stiff':
        # stiff ODEs benefit from adaptive approach
        return AdaptiveODEPINNs(out_dim, hidden_dim, num_layer)
    
    else:  # custom or default
        return EnhancedODEPINNs(out_dim, hidden_dim, num_layer, use_time_embedding=enhanced)

# weight initialization specifically for ODE PINNs
def init_ode_pinn_weights(model):
    """
    Initialize PINN weights for better ODE solving performance
    """
    for m in model.modules():
        if isinstance(m, nn.Linear):
            # xavier initialization with small bias
            nn.init.xavier_uniform_(m.weight, gain=0.5)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.01)

if __name__ == "__main__":
    # test the ODE PINN implementations
    print("Testing ODE PINN implementations")
    print("=" * 40)
    
    # test basic ODE PINN
    basic_pinn = ODEPINNs(out_dim=4, hidden_dim=64, num_layer=3)
    init_ode_pinn_weights(basic_pinn)
    
    # test enhanced ODE PINN
    enhanced_pinn = EnhancedODEPINNs(out_dim=4, hidden_dim=64, num_layer=3)
    init_ode_pinn_weights(enhanced_pinn)
    
    # test adaptive ODE PINN
    adaptive_pinn = AdaptiveODEPINNs(out_dim=4, hidden_dim=64, num_layer=3)
    init_ode_pinn_weights(adaptive_pinn)
    
    # test factory function
    seir_pinn = create_ode_pinn('seir', out_dim=4, enhanced=True)
    
    # test forward pass
    t_test = torch.linspace(0, 10, 50).reshape(-1, 1)
    
    print(f"Basic PINN output shape: {basic_pinn(t_test).shape}")
    print(f"Enhanced PINN output shape: {enhanced_pinn(t_test).shape}")
    print(f"Adaptive PINN output shape: {adaptive_pinn(t_test).shape}")
    print(f"SEIR PINN output shape: {seir_pinn(t_test).shape}")
    
    # count parameters
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nParameter counts:")
    print(f"Basic PINN: {count_parameters(basic_pinn)}")
    print(f"Enhanced PINN: {count_parameters(enhanced_pinn)}")
    print(f"Adaptive PINN: {count_parameters(adaptive_pinn)}")
    print(f"SEIR PINN: {count_parameters(seir_pinn)}")
    
    print("\n✓ All ODE PINN tests passed!")
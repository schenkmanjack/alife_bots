"""
Neural Frame Policy for controlling muscles in algovivo simulation.
"""

import torch
import torch.nn as nn


class NeuralFramePolicy(nn.Module):
    def __init__(self, num_vertices, num_muscles, space_dim=2,
                 hidden_size=32, active=True, stochastic=False, std_dev=0.05,
                 device='cpu'):
        """
        Initialize neural frame policy.
        
        Args:
            num_vertices: Number of vertices
            num_muscles: Number of muscles
            space_dim: Spatial dimension (2 for 2D)
            hidden_size: Hidden layer size
            active: Whether policy is active
            stochastic: Whether to add noise
            std_dev: Standard deviation for noise
            device: Device to run on
        """
        super().__init__()
        self.num_vertices = num_vertices
        self.num_muscles = num_muscles
        self.space_dim = space_dim
        self.active = active
        self.stochastic = stochastic
        self.std_dev = std_dev
        self.device = device
        
        # Input is projected positions and velocities: (num_vertices * space_dim * 2)
        input_size = num_vertices * space_dim * 2
        
        # Neural network
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_muscles),
            nn.Tanh()
        ).to(device)
        
        # Policy parameters (set via load_data)
        self.min_a = None
        self.max_abs_da = None
        self.center_vertex_id = None
        self.forward_vertex_id = None
        self.clockwise = False
        
        # Buffers for projected positions/velocities
        self.register_buffer('projected_pos', torch.zeros(num_vertices, space_dim))
        self.register_buffer('projected_vel', torch.zeros(num_vertices, space_dim))
    
    def load_data(self, data):
        """
        Load policy weights and parameters from data dictionary.
        
        Args:
            data: Dictionary containing:
                - fc1: {weight, bias}
                - fc2: {weight, bias}
                - min_a: minimum muscle action
                - max_abs_da: maximum absolute change in action
                - center_vertex_id: center vertex ID for projection
                - forward_vertex_id: forward vertex ID for projection
        """
        # Load weights
        self.model[0].weight.data = torch.tensor(
            data['fc1']['weight'], device=self.device, dtype=torch.float32
        )
        self.model[0].bias.data = torch.tensor(
            data['fc1']['bias'], device=self.device, dtype=torch.float32
        )
        self.model[2].weight.data = torch.tensor(
            data['fc2']['weight'], device=self.device, dtype=torch.float32
        )
        self.model[2].bias.data = torch.tensor(
            data['fc2']['bias'], device=self.device, dtype=torch.float32
        )
        
        # Load parameters
        self.min_a = data.get('min_a', 0.3)
        self.max_abs_da = data.get('max_abs_da', 0.3)
        self.center_vertex_id = data['center_vertex_id']
        self.forward_vertex_id = data['forward_vertex_id']
    
    def make_neural_policy_input(self, pos, vel):
        """
        Create policy input by projecting positions and velocities relative to center vertex.
        Matches the C++ implementation in frame_projection.h.
        
        Args:
            pos: (num_vertices, space_dim) positions
            vel: (num_vertices, space_dim) velocities
        
        Returns:
            (num_vertices * space_dim * 2,) input vector
        """
        center_pos = pos[self.center_vertex_id]
        forward_pos = pos[self.forward_vertex_id]
        
        # Compute forward direction (normalized)
        forward_dir = forward_pos - center_pos
        forward_norm = torch.norm(forward_dir)
        if forward_norm < 1e-6:
            forward_dir = torch.tensor([1.0, 0.0], device=self.device)
        else:
            forward_dir = forward_dir / forward_norm
        
        # Compute right direction (perpendicular to forward)
        forward_dir_detached = forward_dir.detach()
        if self.clockwise:
            right_dir = torch.tensor([forward_dir_detached[1], -forward_dir_detached[0]], device=self.device)
        else:
            right_dir = torch.tensor([-forward_dir_detached[1], forward_dir_detached[0]], device=self.device)
        
        # Project positions (subtract origin first)
        for i in range(self.num_vertices):
            rel_pos = pos[i] - center_pos  # Subtract origin for positions
            proj_forward = torch.dot(rel_pos, forward_dir)
            proj_right = torch.dot(rel_pos, right_dir)
            self.projected_pos[i, 0] = proj_forward
            self.projected_pos[i, 1] = proj_right
        
        # Project velocities (do NOT subtract origin)
        for i in range(self.num_vertices):
            vel_forward = torch.dot(vel[i], forward_dir)
            vel_right = torch.dot(vel[i], right_dir)
            self.projected_vel[i, 0] = vel_forward
            self.projected_vel[i, 1] = vel_right
        
        # Flatten to input vector: [pos0, pos1, ..., vel0, vel1, ...]
        input_vec = torch.cat([
            self.projected_pos.flatten(),
            self.projected_vel.flatten()
        ]).to(self.device)
        
        return input_vec
    
    def step(self, system, trace=None):
        """
        Perform one policy step, updating muscle actions.
        
        Args:
            system: System instance
            trace: Optional dictionary to store trace information
        
        Returns:
            Dictionary with policy_input and policy_output if trace is provided
        """
        # Create policy input (use system.pos and system.vel, not pos0/vel0)
        policy_input = self.make_neural_policy_input(system.pos, system.vel)
        
        # Forward pass
        with torch.no_grad():
            da = self.model(policy_input)
        
        # Store trace BEFORE processing (matches original: trace stores raw da)
        if trace is not None:
            trace['policy_input'] = policy_input.cpu().numpy().tolist()
            trace['policy_output'] = da.cpu().numpy().tolist()
        
        # Process outputs
        da_processed = da.clone()
        
        if self.active:
            if self.stochastic:
                noise = torch.randn_like(da) * self.std_dev
                da_processed = da_processed + noise
        else:
            da_processed.fill_(1.0)
        
        # Clamp da
        da_processed = torch.clamp(da_processed, -self.max_abs_da, self.max_abs_da)
        
        # Update muscle actions: a = a + da, then clamp
        system.a = system.a + da_processed
        system.a = torch.clamp(system.a, self.min_a, 1.0)
        
        return trace

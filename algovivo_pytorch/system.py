"""
System class for algovivo simulation.
Manages simulation state and performs backward Euler updates.
"""

import torch
import torch.nn as nn
from .energy import backward_euler_loss


class System:
    def __init__(self, space_dim=2, h=0.033, vertex_mass=1.0, g=9.8,
                 k_collision=14000.0, k_friction=300.0, k_muscle=90.0,
                 device='cpu'):
        """
        Initialize simulation system.
        
        Args:
            space_dim: Spatial dimension (2 for 2D)
            h: Timestep
            vertex_mass: Mass per vertex
            g: Gravitational acceleration
            k_collision: Collision stiffness
            k_friction: Friction coefficient
            k_muscle: Muscle spring stiffness
            device: Device to run on ('cpu' or 'cuda')
        """
        self.space_dim = space_dim
        self.h = h
        self.vertex_mass = vertex_mass
        self.g = g
        self.k_collision = k_collision
        self.k_friction = k_friction
        self.k_muscle = k_muscle
        self.device = device
        
        # State variables
        self.pos0 = None  # Previous positions
        self.vel0 = None  # Previous velocities
        self.pos = None   # Current positions (will be optimized)
        self.vel = None   # Current velocities (computed after optimization)
        
        # Mesh data
        self.triangles = None
        self.rsi = None
        self.mu = None
        self.lambda_param = None
        
        # Muscles
        self.muscles = None
        self.l0 = None
        self.a = None  # Muscle actions
        
        self.num_vertices = 0
        self.num_triangles = 0
        self.num_muscles = 0
    
    def set(self, pos, triangles=None, triangles_rsi=None, muscles=None,
            muscles_l0=None, mu=None, lambda_param=None):
        """
        Set mesh and simulation data.
        
        Args:
            pos: (num_vertices, space_dim) initial positions
            triangles: (num_triangles, 3) triangle vertex indices
            triangles_rsi: (num_triangles, 2, 2) reference shape inverse matrices
            muscles: (num_muscles, 2) muscle vertex indices
            muscles_l0: (num_muscles,) muscle rest lengths
            mu: (num_triangles,) Lame parameter mu per triangle
            lambda_param: (num_triangles,) Lame parameter lambda per triangle
        """
        # Convert to tensors if needed
        if not isinstance(pos, torch.Tensor):
            pos = torch.tensor(pos, dtype=torch.float32, device=self.device)
        else:
            pos = pos.to(self.device)
        
        self.num_vertices = pos.shape[0]
        
        # Initialize positions and velocities
        self.pos0 = pos.clone()
        self.vel0 = torch.zeros_like(pos)
        self.pos = pos.clone()
        self.vel = torch.zeros_like(pos)
        
        # Set triangles
        if triangles is not None:
            if not isinstance(triangles, torch.Tensor):
                triangles = torch.tensor(triangles, dtype=torch.long, device=self.device)
            else:
                triangles = triangles.to(self.device)
            self.triangles = triangles
            self.num_triangles = triangles.shape[0]
            
            if triangles_rsi is not None:
                if not isinstance(triangles_rsi, torch.Tensor):
                    # Handle list of 2x2 matrices: [[[a,b],[c,d]], ...]
                    if isinstance(triangles_rsi[0][0], (list, tuple)):
                        # Convert list of 2x2 matrices to tensor
                        triangles_rsi = torch.tensor(triangles_rsi, dtype=torch.float32, device=self.device)
                    else:
                        # Handle flattened format: (num_triangles, 4) or (num_triangles, 2, 2)
                        triangles_rsi = torch.tensor(triangles_rsi, dtype=torch.float32, device=self.device)
                else:
                    triangles_rsi = triangles_rsi.to(self.device)
                # Reshape rsi from (num_triangles, 4) to (num_triangles, 2, 2) if needed
                if triangles_rsi.dim() == 2 and triangles_rsi.shape[1] == 4:
                    self.rsi = triangles_rsi.view(self.num_triangles, 2, 2)
                elif triangles_rsi.dim() == 3 and triangles_rsi.shape[1] == 2 and triangles_rsi.shape[2] == 2:
                    self.rsi = triangles_rsi
                else:
                    raise ValueError(f"Invalid rsi shape: {triangles_rsi.shape}")
            else:
                # Compute rsi from initial positions
                self.rsi = self._compute_rsi(pos, triangles)
            
            # Set material parameters
            if mu is not None:
                if not isinstance(mu, torch.Tensor):
                    mu = torch.tensor(mu, dtype=torch.float32, device=self.device)
                else:
                    mu = mu.to(self.device)
                self.mu = mu
            else:
                # Default values (from Triangles.js: mu.fill_(500))
                self.mu = torch.ones(self.num_triangles, device=self.device) * 500.0
            
            if lambda_param is not None:
                if not isinstance(lambda_param, torch.Tensor):
                    lambda_param = torch.tensor(lambda_param, dtype=torch.float32, device=self.device)
                else:
                    lambda_param = lambda_param.to(self.device)
                self.lambda_param = lambda_param
            else:
                # Default values (from Triangles.js: lambda.fill_(50))
                self.lambda_param = torch.ones(self.num_triangles, device=self.device) * 50.0
        
        # Set muscles
        if muscles is not None:
            if not isinstance(muscles, torch.Tensor):
                muscles = torch.tensor(muscles, dtype=torch.long, device=self.device)
            else:
                muscles = muscles.to(self.device)
            self.muscles = muscles
            self.num_muscles = muscles.shape[0]
            
            if muscles_l0 is not None:
                if not isinstance(muscles_l0, torch.Tensor):
                    muscles_l0 = torch.tensor(muscles_l0, dtype=torch.float32, device=self.device)
                else:
                    muscles_l0 = muscles_l0.to(self.device)
                self.l0 = muscles_l0
            else:
                # Compute l0 from initial positions
                self.l0 = self._compute_l0(pos, muscles)
            
            # Initialize muscle actions to 1.0 (relaxed)
            self.a = torch.ones(self.num_muscles, device=self.device)
        else:
            self.muscles = torch.empty((0, 2), dtype=torch.long, device=self.device)
            self.l0 = torch.empty(0, dtype=torch.float32, device=self.device)
            self.a = torch.empty(0, dtype=torch.float32, device=self.device)
    
    def _compute_rsi(self, pos, triangles):
        """Compute reference shape inverse matrices from initial positions."""
        num_triangles = triangles.shape[0]
        rsi = torch.zeros(num_triangles, 2, 2, device=self.device, dtype=pos.dtype)
        
        for i in range(num_triangles):
            ia, ib, ic = triangles[i]
            a = pos[ia]
            b = pos[ib]
            c = pos[ic]
            
            ab = b - a
            ac = c - a
            
            # Reference shape matrix
            rs = torch.stack([ab, ac], dim=1)  # (2, 2)
            
            # Compute inverse
            det = rs[0, 0] * rs[1, 1] - rs[0, 1] * rs[1, 0]
            if abs(det) > 1e-10:
                inv_det = 1.0 / det
                rsi[i, 0, 0] = rs[1, 1] * inv_det
                rsi[i, 0, 1] = -rs[0, 1] * inv_det
                rsi[i, 1, 0] = -rs[1, 0] * inv_det
                rsi[i, 1, 1] = rs[0, 0] * inv_det
        
        return rsi
    
    def _compute_l0(self, pos, muscles):
        """Compute rest lengths from initial positions."""
        num_muscles = muscles.shape[0]
        l0 = torch.zeros(num_muscles, device=self.device, dtype=pos.dtype)
        
        for i in range(num_muscles):
            i1, i2 = muscles[i]
            p1 = pos[i1]
            p2 = pos[i2]
            l0[i] = torch.norm(p1 - p2)
        
        return l0
    
    def step(self, max_optim_iters=100, grad_q_tol=0.5e-5, step_size_init=1.0,
             backtracking_scale=0.3, verbose=False):
        """
        Perform one simulation step using backward Euler.
        
        Args:
            max_optim_iters: Maximum optimization iterations
            grad_q_tol: Gradient convergence tolerance
            step_size_init: Initial step size for line search
            backtracking_scale: Backtracking factor for line search
            verbose: Whether to print optimization progress
        """
        # Initialize pos to predicted position based on velocity
        self.pos = self.pos0 + self.h * self.vel0
        self.pos.requires_grad_(True)
        
        # Optimization loop
        for iter in range(max_optim_iters):
            # Compute loss
            loss = backward_euler_loss(
                self.pos, self.pos0, self.vel0, self.h,
                self.vertex_mass, self.g, self.k_collision, self.k_friction,
                self.triangles, self.rsi, self.mu, self.lambda_param,
                self.muscles, self.a, self.l0, self.k_muscle
            )
            
            # Compute gradients
            grad = torch.autograd.grad(loss, self.pos, create_graph=False)[0]
            
            # Check convergence
            # Compute max squared gradient magnitude per vertex
            grad_squared = torch.sum(grad * grad, dim=1)  # (num_vertices,)
            grad_max_q = torch.max(grad_squared).item()
            
            if grad_max_q < grad_q_tol:
                if verbose:
                    print(f"  Converged at iteration {iter}, grad_max_q={grad_max_q:.2e}")
                break
            
            # Line search
            step_size = step_size_init
            pos_new = self.pos - step_size * grad
            
            # Evaluate loss at new position
            with torch.no_grad():
                loss_new = backward_euler_loss(
                    pos_new, self.pos0, self.vel0, self.h,
                    self.vertex_mass, self.g, self.k_collision, self.k_friction,
                    self.triangles, self.rsi, self.mu, self.lambda_param,
                    self.muscles, self.a, self.l0, self.k_muscle
                )
            
            # Backtracking line search
            max_line_search_iters = 20
            line_search_iter = 0
            while loss_new.item() >= loss.item() and step_size > 1e-10 and line_search_iter < max_line_search_iters:
                step_size *= backtracking_scale
                pos_new = self.pos - step_size * grad
                with torch.no_grad():
                    loss_new = backward_euler_loss(
                        pos_new, self.pos0, self.vel0, self.h,
                        self.vertex_mass, self.g, self.k_collision, self.k_friction,
                        self.triangles, self.rsi, self.mu, self.lambda_param,
                        self.muscles, self.a, self.l0, self.k_muscle
                    )
                line_search_iter += 1
            
            # Update position
            with torch.no_grad():
                self.pos = pos_new.clone()
                self.pos.requires_grad_(True)
        
        # Update velocities: vel1 = (pos1 - pos0) / h
        with torch.no_grad():
            self.vel = (self.pos - self.pos0) / self.h
            
            # Update state for next step
            self.pos0 = self.pos.clone()
            self.vel0 = self.vel.clone()

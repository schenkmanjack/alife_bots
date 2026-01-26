"""
Energy functions for algovivo simulation.
Implements the six energy functions: Gravity, Collision, Triangles (Neo-Hookean),
Muscles, Friction, and Inertia.
"""

import torch
import torch.nn as nn


def gravity_energy(pos, vertex_mass, g):
    """
    Gravity potential energy: m * g * y
    
    Args:
        pos: (num_vertices, space_dim) tensor of vertex positions
        vertex_mass: scalar mass per vertex
        g: gravitational acceleration
    
    Returns:
        Scalar energy value
    """
    # Extract y-coordinate (index 1 in 2D)
    py = pos[:, 1]
    return torch.sum(py * vertex_mass * g)


def collision_energy(pos, k_collision):
    """
    Collision energy: quadratic penalty for vertices below ground (y < 0)
    
    Args:
        pos: (num_vertices, space_dim) tensor of vertex positions
        k_collision: collision stiffness coefficient
    
    Returns:
        Scalar energy value
    """
    py = pos[:, 1]
    # Only penalize vertices below ground
    below_ground = py < 0
    penalty = torch.where(below_ground, k_collision * py * py, torch.zeros_like(py))
    return torch.sum(penalty)


def triangle_energy(pos, triangles, rsi, mu, lambda_param, w=1.0):
    """
    Neo-Hookean elastic energy for triangular mesh elements.
    Vectorized implementation.
    
    Args:
        pos: (num_vertices, space_dim) tensor of vertex positions
        triangles: (num_triangles, 3) tensor of triangle vertex indices
        rsi: (num_triangles, 2, 2) tensor of reference shape inverse matrices
        mu: (num_triangles,) Lame parameter mu per triangle
        lambda_param: (num_triangles,) Lame parameter lambda per triangle
        w: weight (typically triangle area)
    
    Returns:
        Scalar energy value
    """
    num_triangles = triangles.shape[0]
    if num_triangles == 0:
        return torch.tensor(0.0, device=pos.device, dtype=pos.dtype)
    
    # Get triangle vertices: (num_triangles, 3, 2)
    tri_verts = pos[triangles]  # (num_triangles, 3, 2)
    
    # Compute edge vectors
    a = tri_verts[:, 0]  # (num_triangles, 2)
    b = tri_verts[:, 1]  # (num_triangles, 2)
    c = tri_verts[:, 2]  # (num_triangles, 2)
    
    ab = b - a  # (num_triangles, 2)
    ac = c - a  # (num_triangles, 2)
    
    # Build reference shape matrices: rs = [[abx, acx], [aby, acy]] for each triangle
    # Shape: (num_triangles, 2, 2)
    rs = torch.stack([
        torch.stack([ab[:, 0], ac[:, 0]], dim=1),  # (num_triangles, 2)
        torch.stack([ab[:, 1], ac[:, 1]], dim=1)   # (num_triangles, 2)
    ], dim=1)  # (num_triangles, 2, 2)
    
    # Apply reference shape inverse: F = rs @ rsi
    # Batch matrix multiplication: (num_triangles, 2, 2) @ (num_triangles, 2, 2)
    F = torch.bmm(rs, rsi)  # (num_triangles, 2, 2)
    
    # Compute invariants
    I1 = torch.sum(F * F, dim=(1, 2))  # (num_triangles,) - sum of squares
    J = torch.det(F)  # (num_triangles,)
    
    # Neo-Hookean energy
    qlogJ = -1.5 + 2 * J - 0.5 * J * J
    psi_mu = 0.5 * mu * (I1 - 2) - mu * qlogJ
    psi_lambda = 0.5 * lambda_param * qlogJ * qlogJ
    
    total_energy = w * torch.sum(psi_mu + psi_lambda)
    
    return total_energy


def muscle_energy(pos, muscles, a, l0, k):
    """
    Muscle spring energy with action-dependent rest length.
    Vectorized implementation.
    
    Args:
        pos: (num_vertices, space_dim) tensor of vertex positions
        muscles: (num_muscles, 2) tensor of muscle vertex indices
        a: (num_muscles,) tensor of muscle actions (0.3-1.0, where 1.0 is relaxed)
        l0: (num_muscles,) tensor of rest lengths
        k: scalar spring stiffness
    
    Returns:
        Scalar energy value
    """
    num_muscles = muscles.shape[0]
    if num_muscles == 0:
        return torch.tensor(0.0, device=pos.device, dtype=pos.dtype)
    
    # Get muscle endpoints: (num_muscles, 2, 2)
    muscle_verts = pos[muscles]  # (num_muscles, 2, 2)
    
    # Compute current lengths
    d = muscle_verts[:, 0] - muscle_verts[:, 1]  # (num_muscles, 2)
    l = torch.norm(d, dim=1) + 1e-6  # (num_muscles,)
    
    # Action-dependent rest length
    al0 = a * l0  # (num_muscles,)
    
    # Spring energy: 0.5 * k * (dl/al0)^2
    dl = (l - al0) / al0  # (num_muscles,)
    total_energy = 0.5 * k * torch.sum(dl * dl)
    
    return total_energy


def friction_energy(pos, pos0, h, k_friction):
    """
    Friction energy: penalizes horizontal motion for vertices in contact with ground.
    
    Args:
        pos: (num_vertices, space_dim) tensor of current positions
        pos0: (num_vertices, space_dim) tensor of previous positions
        h: timestep
        k_friction: friction coefficient
    
    Returns:
        Scalar energy value
    """
    eps = 1e-2
    px = pos[:, 0]
    p0x = pos0[:, 0]
    p0y = pos0[:, 1]
    
    # Height relative to ground threshold
    height = p0y - eps
    
    # Only apply friction to vertices below threshold
    below_threshold = height < 0
    
    # Horizontal velocity
    vx = (px - p0x) / h
    
    # Friction energy: k_friction * vx^2 * (-height)
    penalty = torch.where(
        below_threshold,
        k_friction * vx * vx * (-height),
        torch.zeros_like(vx)
    )
    
    return torch.sum(penalty)


def inertia_energy(pos, pos0, vel0, h, vertex_mass):
    """
    Inertial energy: penalizes deviation from predicted position based on velocity.
    Implements Newton's first law within backward Euler framework.
    
    Args:
        pos: (num_vertices, space_dim) tensor of current positions
        pos0: (num_vertices, space_dim) tensor of previous positions
        vel0: (num_vertices, space_dim) tensor of previous velocities
        h: timestep
        vertex_mass: scalar mass per vertex
    
    Returns:
        Scalar energy value
    """
    # Predicted position based on velocity
    y = pos0 + h * vel0
    
    # Deviation from predicted
    d = pos - y
    
    # Energy: m * ||d||^2
    d_squared = torch.sum(d * d, dim=1)
    return torch.sum(vertex_mass * d_squared)


def backward_euler_loss(pos, pos0, vel0, h, vertex_mass, g, k_collision, k_friction,
                        triangles, rsi, mu, lambda_param, muscles, a, l0, k_muscle):
    """
    Total backward Euler loss combining all energy functions.
    
    Loss = 0.5 * inertial_energy + h^2 * potential_energy
    
    Args:
        pos: (num_vertices, space_dim) tensor of current positions (differentiable)
        pos0: (num_vertices, space_dim) tensor of previous positions
        vel0: (num_vertices, space_dim) tensor of previous velocities
        h: timestep
        vertex_mass: scalar mass per vertex
        g: gravitational acceleration
        k_collision: collision stiffness
        k_friction: friction coefficient
        triangles: (num_triangles, 3) triangle indices
        rsi: (num_triangles, 2, 2) reference shape inverse matrices
        mu: (num_triangles,) Lame parameter mu per triangle
        lambda_param: (num_triangles,) Lame parameter lambda per triangle
        muscles: (num_muscles, 2) muscle vertex indices
        a: (num_muscles,) muscle actions
        l0: (num_muscles,) muscle rest lengths
        k_muscle: muscle spring stiffness
    
    Returns:
        Scalar loss value
    """
    # Inertial energy
    inertial_e = inertia_energy(pos, pos0, vel0, h, vertex_mass)
    
    # Potential energies
    gravity_e = gravity_energy(pos, vertex_mass, g)
    collision_e = collision_energy(pos, k_collision)
    triangle_e = triangle_energy(pos, triangles, rsi, mu, lambda_param)
    muscle_e = muscle_energy(pos, muscles, a, l0, k_muscle)
    friction_e = friction_energy(pos, pos0, h, k_friction)
    
    potential_e = gravity_e + collision_e + triangle_e + muscle_e + friction_e
    
    # Total loss
    loss = 0.5 * inertial_e + h * h * potential_e
    
    return loss

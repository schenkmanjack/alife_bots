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


# ============================================================================
# Batched versions of energy functions
# ============================================================================

def create_triangle_mask(triangles, num_vertices_per_system):
    """
    Create a mask indicating which triangles are valid for each system.
    
    A triangle is valid if all its vertices exist in the system (i.e., vertex indices
    are less than the number of real vertices for that system).
    
    Args:
        triangles: [num_triangles, 3] tensor of triangle vertex indices
        num_vertices_per_system: [batch_size] tensor of number of real vertices per system
    
    Returns:
        mask: [batch_size, num_triangles] boolean tensor, True where triangle is valid
    """
    batch_size = num_vertices_per_system.shape[0]
    num_triangles = triangles.shape[0]
    
    # Expand num_vertices_per_system: [batch_size, 1]
    max_vertices = num_vertices_per_system.unsqueeze(1)  # [batch_size, 1]
    
    # Check if all vertices in each triangle are valid: [batch_size, num_triangles]
    # For each triangle, check if all 3 vertices are < num_vertices for that system
    triangle_max_vertex = torch.max(triangles, dim=1)[0]  # [num_triangles] - max vertex index per triangle
    triangle_max_vertex_expanded = triangle_max_vertex.unsqueeze(0)  # [1, num_triangles]
    
    # Triangle is valid if its max vertex index < num_vertices for that system
    mask = triangle_max_vertex_expanded < max_vertices  # [batch_size, num_triangles]
    
    return mask


def create_muscle_mask(muscles, num_vertices_per_system):
    """
    Create a mask indicating which muscles are valid for each system.
    
    A muscle is valid if both its endpoint vertices exist in the system.
    
    Args:
        muscles: [num_muscles, 2] tensor of muscle vertex indices
        num_vertices_per_system: [batch_size] tensor of number of real vertices per system
    
    Returns:
        mask: [batch_size, num_muscles] boolean tensor, True where muscle is valid
    """
    batch_size = num_vertices_per_system.shape[0]
    num_muscles = muscles.shape[0]
    
    # Expand num_vertices_per_system: [batch_size, 1]
    max_vertices = num_vertices_per_system.unsqueeze(1)  # [batch_size, 1]
    
    # Check if both endpoints are valid: [batch_size, num_muscles]
    muscle_max_vertex = torch.max(muscles, dim=1)[0]  # [num_muscles] - max vertex index per muscle
    muscle_max_vertex_expanded = muscle_max_vertex.unsqueeze(0)  # [1, num_muscles]
    
    # Muscle is valid if its max vertex index < num_vertices for that system
    mask = muscle_max_vertex_expanded < max_vertices  # [batch_size, num_muscles]
    
    return mask


def validate_padded_system(pos, vertex_mass, num_vertices_per_system=None):
    """
    Validate that a padded system is set up correctly.
    
    Checks:
    1. Padded vertices (beyond num_vertices_per_system) have zero mass
    2. Padded vertices are at origin (optional check)
    
    Args:
        pos: [batch_size, num_vertices, 2] tensor of positions
        vertex_mass: [batch_size, num_vertices] tensor of masses
        num_vertices_per_system: [batch_size] tensor of real vertex counts, or None to skip validation
    
    Returns:
        is_valid: bool, True if system is properly padded
        warnings: list of warning messages
    """
    warnings = []
    
    if num_vertices_per_system is not None:
        batch_size = pos.shape[0]
        max_vertices = pos.shape[1]
        
        # Ensure vertex_mass has batch dimension
        if vertex_mass.dim() == 1:
            vertex_mass = vertex_mass.unsqueeze(0)
        
        # Check that padded vertices have zero mass
        for b in range(batch_size):
            num_real = num_vertices_per_system[b].item()
            if num_real < max_vertices:
                padded_mass = vertex_mass[b, num_real:]
                if torch.any(padded_mass > 1e-6):
                    warnings.append(f"System {b}: Padded vertices (indices {num_real}-{max_vertices-1}) have non-zero mass")
        
        # Check that padded vertices are at origin (optional, but recommended)
        for b in range(batch_size):
            num_real = num_vertices_per_system[b].item()
            if num_real < max_vertices:
                padded_pos = pos[b, num_real:, :]
                if torch.any(torch.abs(padded_pos) > 1e-3):
                    warnings.append(f"System {b}: Padded vertices are not at origin (may cause issues)")
    
    is_valid = len(warnings) == 0
    return is_valid, warnings

def inertia_energy_batched(pos, pos0, vel0, h, vertex_mass):
    """
    Batched inertia energy: penalizes deviation from predicted position based on velocity.
    
    Args:
        pos: [batch_size, num_vertices, 2] tensor of current positions
        pos0: [batch_size, num_vertices, 2] tensor of previous positions
        vel0: [batch_size, num_vertices, 2] tensor of previous velocities
        h: scalar timestep
        vertex_mass: [batch_size, num_vertices] or [num_vertices] tensor of masses
    
    Returns:
        energy: [batch_size] - energy per system
    """
    # Predicted position based on velocity
    y = pos0 + h * vel0  # [batch_size, num_vertices, 2]
    
    # Deviation from predicted
    d = pos - y  # [batch_size, num_vertices, 2]
    
    # Squared norm per vertex: [batch_size, num_vertices]
    d_squared = torch.sum(d * d, dim=-1)
    
    # Ensure vertex_mass has batch dimension for broadcasting
    if vertex_mass.dim() == 1:
        vertex_mass = vertex_mass.unsqueeze(0)  # [1, num_vertices]
    
    # Energy per vertex: [batch_size, num_vertices]
    energy_per_vertex = vertex_mass * d_squared
    
    # Sum over vertices: [batch_size]
    energy = torch.sum(energy_per_vertex, dim=1)
    
    return energy


def gravity_energy_batched(pos, vertex_mass, g):
    """
    Batched gravity potential energy: m * g * y
    
    Args:
        pos: [batch_size, num_vertices, 2] tensor of vertex positions
        vertex_mass: [batch_size, num_vertices] or [num_vertices] tensor of masses
        g: scalar or [batch_size, 2] or [2] gravitational acceleration
    
    Returns:
        energy: [batch_size] - energy per system
    """
    # Extract y-coordinate (index 1 in 2D)
    py = pos[:, :, 1]  # [batch_size, num_vertices]
    
    # Handle vertex_mass broadcasting
    if vertex_mass.dim() == 1:
        vertex_mass = vertex_mass.unsqueeze(0)  # [1, num_vertices]
    
    # Handle g broadcasting
    if isinstance(g, (int, float)) or (isinstance(g, torch.Tensor) and g.dim() == 0):
        # Scalar g - apply to y coordinate only
        energy_per_vertex = py * vertex_mass * g
    elif isinstance(g, torch.Tensor) and g.dim() == 1 and g.shape[0] == 2:
        # [2] vector - use y component
        gy = g[1]
        energy_per_vertex = py * vertex_mass * gy
    elif isinstance(g, torch.Tensor) and g.dim() == 2 and g.shape[1] == 2:
        # [batch_size, 2] - use y component
        gy = g[:, 1].unsqueeze(1)  # [batch_size, 1]
        energy_per_vertex = py * vertex_mass * gy
    else:
        # Assume scalar
        energy_per_vertex = py * vertex_mass * g
    
    # Sum over vertices: [batch_size]
    energy = torch.sum(energy_per_vertex, dim=1)
    
    return energy


def collision_energy_batched(pos, k_collision):
    """
    Batched collision energy: quadratic penalty for vertices below ground (y < 0)
    
    Args:
        pos: [batch_size, num_vertices, 2] tensor of vertex positions
        k_collision: scalar or [batch_size] collision stiffness coefficient
    
    Returns:
        energy: [batch_size] - energy per system
    """
    py = pos[:, :, 1]  # [batch_size, num_vertices]
    
    # Only penalize vertices below ground
    below_ground = py < 0  # [batch_size, num_vertices]
    
    # Handle k_collision broadcasting
    if isinstance(k_collision, (int, float)) or (isinstance(k_collision, torch.Tensor) and k_collision.dim() == 0):
        k_collision_expanded = k_collision
    elif isinstance(k_collision, torch.Tensor) and k_collision.dim() == 1:
        k_collision_expanded = k_collision.unsqueeze(1)  # [batch_size, 1]
    else:
        k_collision_expanded = k_collision
    
    # Penalty per vertex: [batch_size, num_vertices]
    penalty = torch.where(
        below_ground,
        k_collision_expanded * py * py,
        torch.zeros_like(py)
    )
    
    # Sum over vertices: [batch_size]
    energy = torch.sum(penalty, dim=1)
    
    return energy


def triangle_energy_batched(pos, triangles, rsi, mu, lambda_param, w=1.0, triangle_mask=None):
    """
    Batched Neo-Hookean elastic energy for triangular mesh elements.
    
    Args:
        pos: [batch_size, num_vertices, 2] tensor of vertex positions
        triangles: [num_triangles, 3] tensor of triangle vertex indices (shared)
        rsi: [num_triangles, 2, 2] tensor of reference shape inverse matrices (shared)
        mu: scalar or [num_triangles,] Lame parameter mu per triangle (shared)
        lambda_param: scalar or [num_triangles,] Lame parameter lambda per triangle (shared)
        w: scalar weight (typically triangle area)
        triangle_mask: Optional [batch_size, num_triangles] boolean mask. If provided,
                      triangles with False will contribute zero energy for that system.
                      Use this when systems have different topologies.
    
    Returns:
        energy: [batch_size] - energy per system
    """
    num_triangles = triangles.shape[0]
    batch_size = pos.shape[0]
    
    if num_triangles == 0:
        return torch.zeros(batch_size, device=pos.device, dtype=pos.dtype)
    
    # Get triangle vertices for all batches: [batch_size, num_triangles, 3, 2]
    # Use advanced indexing with broadcasting
    tri_indices = triangles  # [num_triangles, 3]
    
    # Get vertex positions: [batch_size, num_triangles, 3, 2]
    # pos[:, triangles, :] doesn't work, so we use: pos[:, triangles[i], :] for each i
    # More efficient: use gather or manual indexing
    tri_verts = pos[:, tri_indices]  # [batch_size, num_triangles, 3, 2]
    
    # Compute edge vectors for each triangle
    a = tri_verts[:, :, 0, :]  # [batch_size, num_triangles, 2]
    b = tri_verts[:, :, 1, :]  # [batch_size, num_triangles, 2]
    c = tri_verts[:, :, 2, :]  # [batch_size, num_triangles, 2]
    
    ab = b - a  # [batch_size, num_triangles, 2]
    ac = c - a  # [batch_size, num_triangles, 2]
    
    # Build reference shape matrices: rs = [[abx, acx], [aby, acy]] for each triangle
    # Shape: [batch_size, num_triangles, 2, 2]
    # Each rs matrix is: [[ab_x, ac_x], [ab_y, ac_y]]
    rs = torch.stack([
        torch.stack([ab[:, :, 0], ac[:, :, 0]], dim=2),  # [batch_size, num_triangles, 2] - first row
        torch.stack([ab[:, :, 1], ac[:, :, 1]], dim=2)   # [batch_size, num_triangles, 2] - second row
    ], dim=2)  # [batch_size, num_triangles, 2, 2]
    
    # Expand rsi to match batch dimension: [batch_size, num_triangles, 2, 2]
    rsi_expanded = rsi.unsqueeze(0).expand(batch_size, -1, -1, -1)
    
    # Apply reference shape inverse: F = rs @ rsi
    # Batch matrix multiplication: [batch_size, num_triangles, 2, 2] @ [batch_size, num_triangles, 2, 2]
    F = torch.bmm(rs.reshape(batch_size * num_triangles, 2, 2),
                  rsi_expanded.reshape(batch_size * num_triangles, 2, 2))
    F = F.reshape(batch_size, num_triangles, 2, 2)  # [batch_size, num_triangles, 2, 2]
    
    # Compute invariants
    I1 = torch.sum(F * F, dim=(-2, -1))  # [batch_size, num_triangles] - sum of squares
    J = torch.det(F.reshape(batch_size * num_triangles, 2, 2)).reshape(batch_size, num_triangles)  # [batch_size, num_triangles]
    
    # Handle mu and lambda_param broadcasting
    if isinstance(mu, (int, float)) or (isinstance(mu, torch.Tensor) and mu.dim() == 0):
        mu_expanded = mu
    elif isinstance(mu, torch.Tensor) and mu.dim() == 1:
        mu_expanded = mu.unsqueeze(0)  # [1, num_triangles]
    else:
        mu_expanded = mu
    
    if isinstance(lambda_param, (int, float)) or (isinstance(lambda_param, torch.Tensor) and lambda_param.dim() == 0):
        lambda_expanded = lambda_param
    elif isinstance(lambda_param, torch.Tensor) and lambda_param.dim() == 1:
        lambda_expanded = lambda_param.unsqueeze(0)  # [1, num_triangles]
    else:
        lambda_expanded = lambda_param
    
    # Neo-Hookean energy
    qlogJ = -1.5 + 2 * J - 0.5 * J * J  # [batch_size, num_triangles]
    psi_mu = 0.5 * mu_expanded * (I1 - 2) - mu_expanded * qlogJ  # [batch_size, num_triangles]
    psi_lambda = 0.5 * lambda_expanded * qlogJ * qlogJ  # [batch_size, num_triangles]
    
    # Apply mask if provided: zero out invalid triangles
    if triangle_mask is not None:
        # triangle_mask: [batch_size, num_triangles]
        psi_mu = torch.where(triangle_mask, psi_mu, torch.zeros_like(psi_mu))
        psi_lambda = torch.where(triangle_mask, psi_lambda, torch.zeros_like(psi_lambda))
    
    # Sum over triangles: [batch_size]
    total_energy = w * torch.sum(psi_mu + psi_lambda, dim=1)
    
    return total_energy


def muscle_energy_batched(pos, muscles, a, l0, k_muscle, muscle_mask=None):
    """
    Batched muscle spring energy with action-dependent rest length.
    
    Args:
        pos: [batch_size, num_vertices, 2] tensor of vertex positions
        muscles: [num_muscles, 2] or [batch_size, num_muscles, 2] tensor of muscle vertex indices.
                 If [num_muscles, 2], muscles are shared across batch.
                 If [batch_size, num_muscles, 2], each system has its own muscle topology.
        a: [batch_size, num_muscles] tensor of muscle actions
        l0: [batch_size, num_muscles] tensor of rest lengths
        k_muscle: scalar or [batch_size] spring stiffness
        muscle_mask: Optional [batch_size, num_muscles] boolean mask. If provided,
                    muscles with False will contribute zero energy for that system.
                    Use this when systems have different topologies.
    
    Returns:
        energy: [batch_size] - energy per system
    """
    batch_size = pos.shape[0]
    
    # Check if muscles are per-system or shared
    if muscles.dim() == 2:
        # Shared muscles: [num_muscles, 2]
        num_muscles = muscles.shape[0]
        if num_muscles == 0:
            return torch.zeros(batch_size, device=pos.device, dtype=pos.dtype)
        
        # Get muscle endpoints: [batch_size, num_muscles, 2]
        # Use advanced indexing: pos[:, muscles] gives [batch_size, num_muscles, 2, 2]
        muscle_verts = pos[:, muscles]  # [batch_size, num_muscles, 2, 2]
        
        p1 = muscle_verts[:, :, 0, :]  # [batch_size, num_muscles, 2]
        p2 = muscle_verts[:, :, 1, :]  # [batch_size, num_muscles, 2]
    elif muscles.dim() == 3:
        # Per-system muscles: [batch_size, num_muscles, 2]
        num_muscles = muscles.shape[1]
        if num_muscles == 0:
            return torch.zeros(batch_size, device=pos.device, dtype=pos.dtype)
        
        # For per-system muscles, use advanced indexing to get endpoints
        # muscles[b, m] = [v1, v2] -> pos[b, v1, :] and pos[b, v2, :]
        batch_indices = torch.arange(batch_size, device=pos.device)  # [batch_size]
        
        # Get vertex indices for each muscle endpoint
        p1_indices = muscles[:, :, 0]  # [batch_size, num_muscles] - vertex indices for first endpoint
        p2_indices = muscles[:, :, 1]  # [batch_size, num_muscles] - vertex indices for second endpoint
        
        # Use advanced indexing: pos[batch_indices[:, None], p1_indices, :]
        # Create batch index tensor: [batch_size, num_muscles]
        batch_idx = batch_indices.unsqueeze(1).expand(-1, num_muscles)  # [batch_size, num_muscles]
        
        # Get endpoints: [batch_size, num_muscles, 2]
        p1 = pos[batch_idx, p1_indices]  # [batch_size, num_muscles, 2]
        p2 = pos[batch_idx, p2_indices]  # [batch_size, num_muscles, 2]
    else:
        raise ValueError(f"muscles must be 2D [num_muscles, 2] or 3D [batch_size, num_muscles, 2], got shape {muscles.shape}")
    
    # Compute current lengths: [batch_size, num_muscles]
    d = p1 - p2  # [batch_size, num_muscles, 2]
    l = torch.norm(d, dim=-1) + 1e-6  # [batch_size, num_muscles]
    
    # Action-dependent rest length: [batch_size, num_muscles]
    al0 = a * l0  # [batch_size, num_muscles]
    
    # Spring energy: 0.5 * k * (dl/al0)^2
    dl = (l - al0) / al0  # [batch_size, num_muscles]
    
    # Handle k_muscle broadcasting
    if isinstance(k_muscle, (int, float)) or (isinstance(k_muscle, torch.Tensor) and k_muscle.dim() == 0):
        k_expanded = k_muscle
    elif isinstance(k_muscle, torch.Tensor) and k_muscle.dim() == 1:
        k_expanded = k_muscle.unsqueeze(1)  # [batch_size, 1]
    else:
        k_expanded = k_muscle
    
    # Energy per muscle: [batch_size, num_muscles]
    energy_per_muscle = 0.5 * k_expanded * dl * dl
    
    # Apply mask if provided: zero out invalid muscles
    if muscle_mask is not None:
        # muscle_mask: [batch_size, num_muscles]
        energy_per_muscle = torch.where(muscle_mask, energy_per_muscle, torch.zeros_like(energy_per_muscle))
    
    # Sum over muscles: [batch_size]
    energy = torch.sum(energy_per_muscle, dim=1)
    
    return energy


def friction_energy_batched(pos, pos0, h, k_friction):
    """
    Batched friction energy: penalizes horizontal motion for vertices in contact with ground.
    
    Args:
        pos: [batch_size, num_vertices, 2] tensor of current positions
        pos0: [batch_size, num_vertices, 2] tensor of previous positions
        h: scalar timestep
        k_friction: scalar or [batch_size] friction coefficient
    
    Returns:
        energy: [batch_size] - energy per system
    """
    eps = 1e-2
    px = pos[:, :, 0]  # [batch_size, num_vertices]
    p0x = pos0[:, :, 0]  # [batch_size, num_vertices]
    p0y = pos0[:, :, 1]  # [batch_size, num_vertices]
    
    # Height relative to ground threshold
    height = p0y - eps  # [batch_size, num_vertices]
    
    # Only apply friction to vertices below threshold
    below_threshold = height < 0  # [batch_size, num_vertices]
    
    # Horizontal velocity
    vx = (px - p0x) / h  # [batch_size, num_vertices]
    
    # Handle k_friction broadcasting
    if isinstance(k_friction, (int, float)) or (isinstance(k_friction, torch.Tensor) and k_friction.dim() == 0):
        k_friction_expanded = k_friction
    elif isinstance(k_friction, torch.Tensor) and k_friction.dim() == 1:
        k_friction_expanded = k_friction.unsqueeze(1)  # [batch_size, 1]
    else:
        k_friction_expanded = k_friction
    
    # Friction energy: k_friction * vx^2 * (-height)
    penalty = torch.where(
        below_threshold,
        k_friction_expanded * vx * vx * (-height),
        torch.zeros_like(vx)
    )
    
    # Sum over vertices: [batch_size]
    energy = torch.sum(penalty, dim=1)
    
    return energy


def backward_euler_loss_batched(pos, pos0, vel0, h, vertex_mass, g, k_collision, k_friction,
                                 triangles, rsi, mu, lambda_param, muscles, a, l0, k_muscle,
                                 triangle_mask=None, muscle_mask=None):
    """
    Batched total backward Euler loss combining all energy functions.
    
    Loss = 0.5 * inertial_energy + h^2 * potential_energy
    
    Args:
        pos: [batch_size, num_vertices, 2] tensor of current positions (differentiable)
        pos0: [batch_size, num_vertices, 2] tensor of previous positions
        vel0: [batch_size, num_vertices, 2] tensor of previous velocities
        h: scalar timestep
        vertex_mass: [batch_size, num_vertices] or [num_vertices] tensor of masses.
                     Padded vertices should have zero mass.
        g: scalar or [batch_size, 2] or [2] gravitational acceleration
        k_collision: scalar or [batch_size] collision stiffness
        k_friction: scalar or [batch_size] friction coefficient
        triangles: [num_triangles, 3] triangle indices (shared across batch)
        rsi: [num_triangles, 2, 2] reference shape inverse matrices (shared)
        mu: scalar or [num_triangles,] Lame parameter mu per triangle (shared)
        lambda_param: scalar or [num_triangles,] Lame parameter lambda per triangle (shared)
        muscles: [num_muscles, 2] or [batch_size, num_muscles, 2] muscle vertex indices.
                 If [num_muscles, 2], muscles are shared across batch (same topology).
                 If [batch_size, num_muscles, 2], each system has its own muscle topology.
                 Use per-system muscles when systems have different muscle geometries.
        a: [batch_size, num_muscles] muscle actions
        l0: [batch_size, num_muscles] muscle rest lengths
        k_muscle: scalar or [batch_size] muscle spring stiffness
        triangle_mask: Optional [batch_size, num_triangles] boolean mask for valid triangles.
                       Use when systems have different topologies. See create_triangle_mask().
        muscle_mask: Optional [batch_size, num_muscles] boolean mask for valid muscles.
                     Use when systems have different topologies. See create_muscle_mask().
                     Note: If using per-system muscles [batch_size, num_muscles, 2], 
                     muscle_mask may not be needed as each system already has its own topology.
    
    Returns:
        loss: [batch_size] - loss per system
    
    Notes:
        For systems with different numbers of vertices:
        1. Pad all systems to the same max size (e.g., 8 vertices)
        2. Set vertex_mass to zero for padded vertices
        3. Set padded vertex positions to origin (0, 0)
        4. Provide triangle_mask and/or muscle_mask to exclude invalid elements
        5. Use create_triangle_mask() and create_muscle_mask() helpers
        
        For systems with different muscle geometries (different vertex connections):
        - Option 1: Use shared muscles [num_muscles, 2] with muscle_mask to mark invalid muscles
        - Option 2: Use per-system muscles [batch_size, num_muscles, 2] where each system
          has its own muscle topology. This is more efficient when topologies differ significantly.
    """
    # Inertial energy
    inertial_e = inertia_energy_batched(pos, pos0, vel0, h, vertex_mass)  # [batch_size]
    
    # Potential energies
    gravity_e = gravity_energy_batched(pos, vertex_mass, g)  # [batch_size]
    collision_e = collision_energy_batched(pos, k_collision)  # [batch_size]
    triangle_e = triangle_energy_batched(pos, triangles, rsi, mu, lambda_param, triangle_mask=triangle_mask)  # [batch_size]
    muscle_e = muscle_energy_batched(pos, muscles, a, l0, k_muscle, muscle_mask=muscle_mask)  # [batch_size]
    friction_e = friction_energy_batched(pos, pos0, h, k_friction)  # [batch_size]
    
    potential_e = gravity_e + collision_e + triangle_e + muscle_e + friction_e  # [batch_size]
    
    # Total loss: [batch_size]
    loss = 0.5 * inertial_e + h * h * potential_e
    
    return loss

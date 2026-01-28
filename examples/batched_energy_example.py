"""
Example: Using batched energy functions with systems of different morphologies.

This demonstrates how to:
1. Pad systems to the same size
2. Create masks for valid triangles/muscles
3. Use batched energy functions with masking
"""

import torch
from algovivo_pytorch.energy import (
    backward_euler_loss_batched,
    create_triangle_mask,
    create_muscle_mask,
    validate_padded_system
)


def prepare_batched_systems(systems_data, max_vertices=8, max_muscles=12):
    """
    Prepare a batch of systems with different morphologies for batched computation.
    
    Args:
        systems_data: List of dicts, each containing:
            - 'pos': [num_vertices, 2] positions
            - 'triangles': [num_triangles, 3] triangle indices
            - 'muscles': [num_muscles, 2] muscle indices
            - 'num_vertices': int, number of real vertices
            - Other system parameters...
        max_vertices: Maximum number of vertices to pad to
        max_muscles: Maximum number of muscles to pad to
    
    Returns:
        Batched tensors and masks ready for backward_euler_loss_batched()
    """
    batch_size = len(systems_data)
    
    # Pad positions to max_vertices
    pos_list = []
    vertex_mass_list = []
    num_vertices_list = []
    
    for sys in systems_data:
        num_real = sys['num_vertices']
        pos_real = sys['pos']
        
        # Pad positions to max_vertices (pad with zeros at origin)
        num_pad = max_vertices - num_real
        if num_pad > 0:
            pos_padded = torch.cat([pos_real, torch.zeros(num_pad, 2)], dim=0)
        else:
            pos_padded = pos_real
        
        # Create vertex mass: 1.0 for real vertices, 0.0 for padded
        mass_real = torch.ones(num_real)
        if num_pad > 0:
            mass_padded = torch.cat([mass_real, torch.zeros(num_pad)], dim=0)
        else:
            mass_padded = mass_real
        
        pos_list.append(pos_padded)
        vertex_mass_list.append(mass_padded)
        num_vertices_list.append(num_real)
    
    # Stack into batches
    pos_batched = torch.stack(pos_list, dim=0)  # [batch_size, max_vertices, 2]
    vertex_mass_batched = torch.stack(vertex_mass_list, dim=0)  # [batch_size, max_vertices]
    num_vertices_per_system = torch.tensor(num_vertices_list)  # [batch_size]
    
    # Find max triangles and muscles across all systems
    # Note: triangles and muscles are SHARED across batch (same topology)
    # They reference vertex indices, which are padded to max_vertices
    all_triangles = systems_data[0]['triangles']
    all_muscles = systems_data[0]['muscles']
    
    # Use the union of all triangles/muscles (or the max set)
    # In practice, all systems should share the same topology
    triangles = all_triangles
    muscles = all_muscles
    
    num_muscles = muscles.shape[0]
    
    # Pad activations and rest lengths to match number of muscles
    # (muscles are shared, so all systems have same num_muscles)
    a_list = []
    l0_list = []
    
    for sys in systems_data:
        a_real = sys.get('a', torch.ones(num_muscles))
        l0_real = sys.get('l0', torch.ones(num_muscles))
        
        # Ensure they match the number of muscles
        if a_real.shape[0] < num_muscles:
            a_real = torch.cat([a_real, torch.ones(num_muscles - a_real.shape[0])], dim=0)
        if l0_real.shape[0] < num_muscles:
            l0_real = torch.cat([l0_real, torch.ones(num_muscles - l0_real.shape[0])], dim=0)
        
        a_list.append(a_real[:num_muscles])
        l0_list.append(l0_real[:num_muscles])
    
    a_batched = torch.stack(a_list, dim=0)  # [batch_size, num_muscles]
    l0_batched = torch.stack(l0_list, dim=0)  # [batch_size, num_muscles]
    
    # Create masks for valid triangles and muscles
    triangle_mask = create_triangle_mask(triangles, num_vertices_per_system)
    muscle_mask = create_muscle_mask(muscles, num_vertices_per_system)
    
    # Validate padding
    is_valid, warnings = validate_padded_system(
        pos_batched, vertex_mass_batched, num_vertices_per_system
    )
    if not is_valid:
        print("⚠ Warnings:", warnings)
    
    return {
        'pos_batched': pos_batched,
        'pos0_batched': pos_batched.clone(),  # In real usage, use previous positions
        'vel0_batched': torch.zeros_like(pos_batched),
        'vertex_mass_batched': vertex_mass_batched,
        'triangles': triangles,
        'muscles': muscles,
        'a_batched': a_batched,
        'l0_batched': l0_batched,
        'triangle_mask': triangle_mask,
        'muscle_mask': muscle_mask,
        'num_vertices_per_system': num_vertices_per_system,
    }


def example_usage():
    """Example of using batched energy functions."""
    
    # Create example systems with different morphologies
    systems_data = [
        {
            'pos': torch.randn(5, 2),  # 5 vertices
            'triangles': torch.tensor([[0, 1, 2], [1, 2, 3], [2, 3, 4]]),
            'muscles': torch.tensor([[0, 1], [1, 2], [2, 3]]),
            'num_vertices': 5,
            'a': torch.ones(3),
            'l0': torch.ones(3),
        },
        {
            'pos': torch.randn(3, 2),  # 3 vertices
            'triangles': torch.tensor([[0, 1, 2], [1, 2, 3], [2, 3, 4]]),  # Same topology
            'muscles': torch.tensor([[0, 1], [1, 2], [2, 3]]),  # Same topology
            'num_vertices': 3,
            'a': torch.ones(3),
            'l0': torch.ones(3),
        },
        {
            'pos': torch.randn(7, 2),  # 7 vertices
            'triangles': torch.tensor([[0, 1, 2], [1, 2, 3], [2, 3, 4]]),
            'muscles': torch.tensor([[0, 1], [1, 2], [2, 3]]),
            'num_vertices': 7,
            'a': torch.ones(3),
            'l0': torch.ones(3),
        },
    ]
    
    # Prepare batched data
    batched = prepare_batched_systems(systems_data, max_vertices=8, max_muscles=12)
    
    # Set up other parameters
    h = 0.033
    g = 9.8
    k_collision = 14000.0
    k_friction = 300.0
    k_muscle = 90.0
    
    # Create dummy rsi, mu, lambda (in practice, these come from your systems)
    num_triangles = batched['triangles'].shape[0]
    rsi = torch.randn(num_triangles, 2, 2)
    mu = torch.ones(num_triangles) * 500.0
    lambda_param = torch.ones(num_triangles) * 50.0
    
    # Compute batched loss
    loss = backward_euler_loss_batched(
        batched['pos_batched'],
        batched['pos0_batched'],
        batched['vel0_batched'],
        h,
        batched['vertex_mass_batched'],
        g,
        k_collision,
        k_friction,
        batched['triangles'],
        rsi,
        mu,
        lambda_param,
        batched['muscles'],
        batched['a_batched'],
        batched['l0_batched'],
        k_muscle,
        triangle_mask=batched['triangle_mask'],
        muscle_mask=batched['muscle_mask'],
    )
    
    print(f"Computed loss for {len(systems_data)} systems:")
    print(f"Loss shape: {loss.shape}")
    print(f"Loss values: {loss}")
    print(f"\nTriangle mask:\n{batched['triangle_mask'].int()}")
    print(f"\nMuscle mask:\n{batched['muscle_mask'].int()}")
    
    return loss


if __name__ == '__main__':
    example_usage()

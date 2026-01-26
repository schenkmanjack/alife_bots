"""
Test trajectory replication from original algovivo repo.
"""

import json
import os
import numpy as np
import torch
from pathlib import Path
from algovivo_pytorch import System, NeuralFramePolicy


def load_json(filename):
    """Load JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)


def to_numpy(tensor):
    """Convert tensor to numpy array."""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return tensor


def compare_arrays(arr1, arr2, rtol=1e-3, atol=1e-5):
    """Compare two arrays with tolerance."""
    arr1 = np.asarray(arr1)
    arr2 = np.asarray(arr2)
    
    if arr1.shape != arr2.shape:
        return False, f"Shape mismatch: {arr1.shape} vs {arr2.shape}"
    
    if not np.allclose(arr1, arr2, rtol=rtol, atol=atol):
        max_diff = np.max(np.abs(arr1 - arr2))
        mean_diff = np.mean(np.abs(arr1 - arr2))
        return False, f"Value mismatch: max_diff={max_diff}, mean_diff={mean_diff}"
    
    return True, ""


def test_trajectory():
    """Test trajectory replication."""
    # Paths
    repo_path = Path(__file__).parent / "algovivo.repo"
    data_dir = repo_path / "test" / "nn" / "data"
    mesh_file = data_dir / "mesh.json"
    policy_file = data_dir / "policy.json"
    trajectory_dir = data_dir / "trajectory"
    
    # Load data
    print("Loading mesh and policy data...")
    mesh_data = load_json(mesh_file)
    policy_data = load_json(policy_file)
    
    # Create system
    print("Creating system...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    system = System(
        space_dim=2,
        h=0.033,
        vertex_mass=1.0,
        g=9.8,
        k_collision=14000.0,  # Match original default
        k_friction=300.0,      # Match original default
        k_muscle=90.0,         # Match original default
        device=device
    )
    
    # Set mesh data
    system.set(
        pos=mesh_data['pos'],
        triangles=mesh_data['triangles'],
        triangles_rsi=mesh_data['rsi'],
        muscles=mesh_data['muscles'],
        muscles_l0=mesh_data['l0']
    )
    
    # Create policy
    print("Creating policy...")
    policy = NeuralFramePolicy(
        num_vertices=system.num_vertices,
        num_muscles=system.num_muscles,
        space_dim=2,
        active=True,
        stochastic=False,
        device=device
    )
    policy.load_data(policy_data)
    
    # Load trajectory files
    trajectory_files = sorted(trajectory_dir.glob("*.json"), key=lambda x: int(x.stem))
    num_steps = len(trajectory_files)
    print(f"Found {num_steps} trajectory steps")
    
    # Run simulation
    print("Running simulation...")
    errors = []
    
    for step_idx in range(num_steps):
        # Load reference data
        ref_data = load_json(trajectory_files[step_idx])
        
        # Set initial state
        system.pos0 = torch.tensor(ref_data['pos0'], device=device, dtype=torch.float32)
        system.vel0 = torch.tensor(ref_data['vel0'], device=device, dtype=torch.float32)
        system.pos = system.pos0.clone()  # Initialize pos
        system.vel = system.vel0.clone()  # Initialize vel
        system.a = torch.tensor(ref_data['a0'], device=device, dtype=torch.float32)
        
        # Policy step (uses system.pos and system.vel)
        policy_trace = {}
        policy.step(system, trace=policy_trace)
        
        # System step
        system.step(max_optim_iters=100)  # Use full iterations for accuracy
        
        # Compare results
        pos1_actual = to_numpy(system.pos0)
        vel1_actual = to_numpy(system.vel0)
        a1_actual = to_numpy(system.a)
        
        pos1_ref = np.array(ref_data['pos1'])
        vel1_ref = np.array(ref_data['vel1'])
        a1_ref = np.array(ref_data['a1'])
        
        # Compare positions
        pos_match, pos_msg = compare_arrays(pos1_actual, pos1_ref, rtol=1e-2, atol=1e-3)
        if not pos_match:
            errors.append(f"Step {step_idx}: Position mismatch - {pos_msg}")
        
        # Compare velocities
        vel_match, vel_msg = compare_arrays(vel1_actual, vel1_ref, rtol=1e-2, atol=1e-3)
        if not vel_match:
            errors.append(f"Step {step_idx}: Velocity mismatch - {vel_msg}")
        
        # Compare actions
        a_match, a_msg = compare_arrays(a1_actual, a1_ref, rtol=1e-2, atol=1e-3)
        if not a_match:
            errors.append(f"Step {step_idx}: Action mismatch - {a_msg}")
        
        # Compare policy input/output (optional, for debugging)
        if step_idx == 0:
            policy_input_ref = np.array(ref_data['policy_input'])
            policy_output_ref = np.array(ref_data['policy_output'])
            
            policy_input_match, policy_input_msg = compare_arrays(
                policy_trace['policy_input'], policy_input_ref, rtol=1e-2, atol=1e-3
            )
            if not policy_input_match:
                errors.append(f"Step {step_idx}: Policy input mismatch - {policy_input_msg}")
            
            policy_output_match, policy_output_msg = compare_arrays(
                policy_trace['policy_output'], policy_output_ref, rtol=1e-2, atol=1e-3
            )
            if not policy_output_match:
                errors.append(f"Step {step_idx}: Policy output mismatch - {policy_output_msg}")
        
        if (step_idx + 1) % 10 == 0:
            print(f"  Completed {step_idx + 1}/{num_steps} steps")
    
    # Report results
    print("\n" + "="*60)
    if errors:
        print(f"FAILED: Found {len(errors)} errors")
        for error in errors[:10]:  # Show first 10 errors
            print(f"  {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
        return False
    else:
        print("SUCCESS: All trajectory steps match reference!")
        return True


if __name__ == "__main__":
    success = test_trajectory()
    exit(0 if success else 1)

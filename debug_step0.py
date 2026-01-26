"""Debug first step to understand differences."""
import json
import numpy as np
import torch
from algovivo_pytorch import System, NeuralFramePolicy
from pathlib import Path

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load data
repo_path = Path("algovivo.repo")
data_dir = repo_path / "test" / "nn" / "data"
mesh_data = json.load(open(data_dir / "mesh.json"))
policy_data = json.load(open(data_dir / "policy.json"))
ref_data = json.load(open(data_dir / "trajectory" / "0.json"))

# Create system
system = System(device=device)
system.set(
    pos=mesh_data['pos'],
    triangles=mesh_data['triangles'],
    triangles_rsi=mesh_data['rsi'],
    muscles=mesh_data['muscles'],
    muscles_l0=mesh_data['l0']
)

# Create policy
policy = NeuralFramePolicy(
    num_vertices=system.num_vertices,
    num_muscles=system.num_muscles,
    device=device
)
policy.load_data(policy_data)

# Set initial state
system.pos0 = torch.tensor(ref_data['pos0'], device=device, dtype=torch.float32)
system.vel0 = torch.tensor(ref_data['vel0'], device=device, dtype=torch.float32)
system.pos = system.pos0.clone()
system.vel = system.vel0.clone()
system.a = torch.tensor(ref_data['a0'], device=device, dtype=torch.float32)

print(f"\nInitial state:")
print(f"  pos0[0] = {system.pos0[0].cpu().numpy()}")
print(f"  vel0[0] = {system.vel0[0].cpu().numpy()}")
print(f"  a[0] = {system.a[0].item()}")

# Policy step
policy_trace = {}
policy.step(system, trace=policy_trace)

print(f"\nAfter policy:")
print(f"  a[0] = {system.a[0].item()}")
print(f"  Expected a[0] = {ref_data['a1'][0]}")

# System step with verbose output
print(f"\nRunning system step...")
system.step(max_optim_iters=100, verbose=True)

print(f"\nAfter system step:")
print(f"  pos[0] = {system.pos0[0].cpu().numpy()}")
print(f"  vel[0] = {system.vel0[0].cpu().numpy()}")
print(f"  Expected pos1[0] = {np.array(ref_data['pos1'][0])}")
print(f"  Expected vel1[0] = {np.array(ref_data['vel1'][0])}")

# Compare
pos_diff = np.abs(system.pos0[0].cpu().numpy() - np.array(ref_data['pos1'][0]))
vel_diff = np.abs(system.vel0[0].cpu().numpy() - np.array(ref_data['vel1'][0]))
print(f"\nDifferences:")
print(f"  pos diff[0] = {pos_diff}")
print(f"  vel diff[0] = {vel_diff}")

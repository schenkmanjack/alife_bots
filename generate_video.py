"""
Generate video visualization of the algovivo simulation.
"""

import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
from algovivo_pytorch import System, NeuralFramePolicy


def load_json(filename):
    """Load JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)


def generate_video(output_filename='simulation.mp4', num_steps=100, fps=30):
    """Generate video of simulation."""
    # Paths
    repo_path = Path("algovivo.repo")
    data_dir = repo_path / "test" / "nn" / "data"
    mesh_file = data_dir / "mesh.json"
    policy_file = data_dir / "policy.json"
    
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
        k_collision=14000.0,
        k_friction=300.0,
        k_muscle=90.0,
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
    
    # Store trajectory (positions and muscle activations)
    trajectory = []
    muscle_activations = []
    
    print(f"Running simulation for {num_steps} steps...")
    for step_idx in range(num_steps):
        # Policy step
        policy.step(system)
        
        # System step
        system.step(max_optim_iters=100)
        
        # Store state
        pos = system.pos0.cpu().numpy()
        a = system.a.cpu().numpy()
        trajectory.append(pos.copy())
        muscle_activations.append(a.copy())
        
        if (step_idx + 1) % 10 == 0:
            print(f"  Completed {step_idx + 1}/{num_steps} steps")
    
    # Create visualization
    print("Creating visualization...")
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Set up plot
    ax.set_xlim(-1, 5)
    ax.set_ylim(-1, 3)
    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Algovivo Simulation')
    ax.grid(True, alpha=0.3)
    
    # Draw ground
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=2, label='Ground')
    
    # Convert triangles to numpy
    triangles = np.array(mesh_data['triangles'])
    muscles = np.array(mesh_data['muscles'])
    
    # Initialize plot elements
    triangle_patches = []
    for _ in range(len(triangles)):
        tri = plt.Polygon([[0, 0], [0, 0], [0, 0]], 
                         facecolor='lightblue', 
                         edgecolor='black', 
                         linewidth=1,
                         alpha=0.6)
        ax.add_patch(tri)
        triangle_patches.append(tri)
    
    muscle_lines = []
    for _ in range(len(muscles)):
        # Muscles will be colored based on activation: red (active) to pink (inactive)
        line, = ax.plot([], [], linewidth=3, alpha=0.8)
        muscle_lines.append(line)
    
    vertex_points, = ax.plot([], [], 'ko', markersize=4)
    
    def animate(frame):
        """Animation function."""
        pos = trajectory[frame]
        a = muscle_activations[frame]
        
        # Update triangles
        for i, tri_idx in enumerate(triangles):
            tri_verts = pos[tri_idx]
            triangle_patches[i].set_xy(tri_verts)
        
        # Update muscles with color based on activation
        # a values range from ~0.3 (contracted) to 1.0 (relaxed)
        # Map to color: red (active/contracted) -> pink (inactive/relaxed)
        for i, muscle_idx in enumerate(muscles):
            p1 = pos[muscle_idx[0]]
            p2 = pos[muscle_idx[1]]
            muscle_lines[i].set_data([p1[0], p2[0]], [p1[1], p2[1]])
            
            # Color interpolation: a=0.3 (contracted) -> red, a=1.0 (relaxed) -> pink
            # Normalize a from [0.3, 1.0] to [0, 1] for interpolation
            normalized_a = (a[i] - 0.3) / (1.0 - 0.3) if a[i] > 0.3 else 0.0
            normalized_a = max(0.0, min(1.0, normalized_a))
            
            # Interpolate between red (255, 0, 0) and pink (250, 190, 190)
            r = int(255 * (1 - normalized_a) + 250 * normalized_a)
            g = int(0 * (1 - normalized_a) + 190 * normalized_a)
            b = int(0 * (1 - normalized_a) + 190 * normalized_a)
            
            muscle_lines[i].set_color((r/255, g/255, b/255))
        
        # Update vertices
        vertex_points.set_data(pos[:, 0], pos[:, 1])
        
        ax.set_title(f'Algovivo Simulation - Step {frame}/{num_steps-1}')
        
        return triangle_patches + muscle_lines + [vertex_points]
    
    # Create animation
    print("Rendering animation...")
    anim = animation.FuncAnimation(
        fig, animate, frames=num_steps,
        interval=1000/fps, blit=True, repeat=True
    )
    
    # Save video
    print(f"Saving video to {output_filename}...")
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps, metadata=dict(artist='Algovivo'), bitrate=1800)
    anim.save(output_filename, writer=writer)
    
    print(f"Video saved to {output_filename}")
    plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', '-o', default='simulation.mp4', help='Output video filename')
    parser.add_argument('--steps', type=int, default=100, help='Number of simulation steps')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second')
    args = parser.parse_args()
    
    generate_video(args.output, args.steps, args.fps)

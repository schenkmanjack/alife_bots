"""
Example script showing how to create robots with different topologies.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from algovivo_pytorch import System


def create_simple_robot():
    """Create a simple 3-vertex robot (like the README example)."""
    print("Creating simple 3-vertex robot...")
    
    # 3 vertices forming a triangle
    pos = [
        [0, 0],    # vertex 0
        [2, 0],    # vertex 1
        [1, 1]     # vertex 2
    ]
    
    # One triangle
    triangles = [
        [0, 1, 2]
    ]
    
    # Two muscles connecting to vertex 2
    muscles = [
        [0, 2],  # muscle from vertex 0 to 2
        [1, 2]   # muscle from vertex 1 to 2
    ]
    
    return pos, triangles, muscles


def create_quadruped():
    """Create a simple quadruped robot."""
    print("Creating quadruped robot...")
    
    # Body and 4 legs
    pos = [
        # Body (center)
        [2, 2],   # 0: body center
        [1, 2],   # 1: body left
        [3, 2],   # 2: body right
        [2, 1],   # 3: body bottom
        
        # Front left leg
        [1, 1],   # 4: front left hip
        [0.5, 0], # 5: front left foot
        
        # Front right leg
        [3, 1],   # 6: front right hip
        [3.5, 0], # 7: front right foot
        
        # Back left leg
        [1, 3],   # 8: back left hip
        [0.5, 4], # 9: back left foot
        
        # Back right leg
        [3, 3],   # 10: back right hip
        [3.5, 4], # 11: back right foot
    ]
    
    # Triangles forming the body and legs
    triangles = [
        # Body
        [0, 1, 3],
        [0, 2, 3],
        [0, 1, 8],
        [0, 2, 10],
        
        # Front left leg
        [4, 1, 3],
        [4, 5, 1],
        
        # Front right leg
        [6, 2, 3],
        [6, 7, 2],
        
        # Back left leg
        [8, 1, 0],
        [8, 9, 1],
        
        # Back right leg
        [10, 2, 0],
        [10, 11, 2],
    ]
    
    # Muscles for locomotion
    muscles = [
        # Body muscles
        [0, 1],
        [0, 2],
        [0, 3],
        
        # Front left leg muscles
        [4, 5],
        [1, 4],
        
        # Front right leg muscles
        [6, 7],
        [2, 6],
        
        # Back left leg muscles
        [8, 9],
        [1, 8],
        
        # Back right leg muscles
        [10, 11],
        [2, 10],
    ]
    
    return pos, triangles, muscles


def create_snake():
    """Create a snake-like robot."""
    print("Creating snake robot...")
    
    num_segments = 8
    segment_length = 0.5
    segment_width = 0.3
    
    pos = []
    triangles = []
    muscles = []
    
    # Create segments
    for i in range(num_segments):
        x = i * segment_length
        y = 1.0
        
        # Each segment has 3 vertices (forming a triangle)
        base_idx = i * 3
        pos.extend([
            [x, y],                    # top vertex
            [x, y - segment_width],    # bottom left
            [x + segment_length, y - segment_width],  # bottom right
        ])
        
        # Triangle for this segment
        triangles.append([base_idx, base_idx + 1, base_idx + 2])
        
        # Muscles connecting segments
        if i > 0:
            # Connect to previous segment
            muscles.append([(i-1)*3, base_idx])
            muscles.append([(i-1)*3 + 1, base_idx + 1])
    
    return pos, triangles, muscles


def create_star():
    """Create a star-shaped robot."""
    print("Creating star robot...")
    
    center = [2, 2]
    radius = 1.0
    num_points = 6
    
    pos = [center]  # Center vertex
    
    # Create star points
    for i in range(num_points):
        angle = 2 * np.pi * i / num_points
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        pos.append([x, y])
    
    # Create triangles from center to adjacent points
    triangles = []
    for i in range(num_points):
        next_i = (i + 1) % num_points
        triangles.append([0, i + 1, next_i + 1])
    
    # Muscles from center to each point
    muscles = []
    for i in range(num_points):
        muscles.append([0, i + 1])
    
    # Muscles around the perimeter
    for i in range(num_points):
        next_i = (i + 1) % num_points
        muscles.append([i + 1, next_i + 1])
    
    return pos, triangles, muscles


def simulate_robot(pos, triangles, muscles, num_steps=100, title="Robot"):
    """Simulate a robot and visualize it."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create system
    system = System(device=device)
    system.set(
        pos=pos,
        triangles=triangles,
        muscles=muscles
    )
    
    print(f"  Created {system.num_vertices} vertices, {system.num_triangles} triangles, {system.num_muscles} muscles")
    
    # Simple periodic muscle control
    trajectory = []
    for step in range(num_steps):
        # Simple periodic control: alternate muscle activations
        t = step * 0.1
        a = torch.ones(system.num_muscles, device=device)
        for i in range(system.num_muscles):
            # Create a wave pattern
            a[i] = 0.5 + 0.5 * np.sin(t + i * 0.5)
        system.a = a
        
        system.step(max_optim_iters=50)
        trajectory.append(system.pos0.cpu().numpy().copy())
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    for frame_idx, frame_num in enumerate([0, num_steps//4, num_steps//2, num_steps-1]):
        ax = axes[frame_idx]
        pos_frame = trajectory[frame_num]
        
        # Draw triangles
        for tri in triangles:
            tri_verts = pos_frame[tri]
            tri_poly = plt.Polygon(tri_verts, facecolor='lightblue', 
                                  edgecolor='black', linewidth=1, alpha=0.6)
            ax.add_patch(tri_poly)
        
        # Draw muscles
        for muscle in muscles:
            p1 = pos_frame[muscle[0]]
            p2 = pos_frame[muscle[1]]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-', linewidth=2, alpha=0.7)
        
        # Draw vertices
        ax.plot(pos_frame[:, 0], pos_frame[:, 1], 'ko', markersize=4)
        
        ax.set_xlim(-1, 5)
        ax.set_ylim(-1, 5)
        ax.set_aspect('equal')
        ax.set_title(f'{title} - Step {frame_num}')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    
    plt.tight_layout()
    filename = f'{title.lower().replace(" ", "_")}_simulation.png'
    plt.savefig(filename)
    print(f"  Saved visualization to {filename}")
    plt.close()


if __name__ == "__main__":
    print("="*60)
    print("Creating robots with different topologies")
    print("="*60)
    
    # Create and simulate different robots
    robots = [
        ("Simple", create_simple_robot()),
        ("Quadruped", create_quadruped()),
        ("Snake", create_snake()),
        ("Star", create_star()),
    ]
    
    for name, (pos, triangles, muscles) in robots:
        print(f"\n{name} Robot:")
        simulate_robot(pos, triangles, muscles, num_steps=50, title=name)
    
    print("\n" + "="*60)
    print("All robots created successfully!")
    print("="*60)

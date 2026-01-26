"""
Generate and visualize random simple robot topologies.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from algovivo_pytorch import System
import random


def validate_triangle(pos, triangle):
    """Validate that a triangle is well-formed (non-degenerate, positive area)."""
    a, b, c = pos[triangle[0]], pos[triangle[1]], pos[triangle[2]]
    ab = np.array(b) - np.array(a)
    ac = np.array(c) - np.array(a)
    # Compute signed area (2D cross product)
    area = ab[0] * ac[1] - ab[1] * ac[0]
    return abs(area) > 1e-4  # Minimum area threshold


def generate_random_simple_robot(num_vertices=5, seed=None):
    """
    Generate a random simple robot topology with validation.
    
    Args:
        num_vertices: Number of vertices (3-10 recommended)
        seed: Random seed for reproducibility
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    print(f"Generating random robot with {num_vertices} vertices...")
    
    # Generate random vertex positions in a reasonable range
    # Ensure vertices are well-spaced to avoid degenerate triangles
    pos = []
    min_distance = 0.3  # Minimum distance between vertices
    
    for i in range(num_vertices):
        max_attempts = 100
        for attempt in range(max_attempts):
            # Create vertices in a roughly circular or blob-like arrangement
            angle = 2 * np.pi * i / num_vertices + np.random.uniform(-0.2, 0.2)
            radius = 0.8 + np.random.uniform(-0.15, 0.15)
            x = 2.0 + radius * np.cos(angle) + np.random.uniform(-0.05, 0.05)
            y = 1.5 + radius * np.sin(angle) + np.random.uniform(-0.05, 0.05)
            
            # Check minimum distance from existing vertices
            too_close = False
            for existing_pos in pos:
                dist = np.sqrt((x - existing_pos[0])**2 + (y - existing_pos[1])**2)
                if dist < min_distance:
                    too_close = True
                    break
            
            if not too_close:
                pos.append([x, y])
                break
        
        if len(pos) == i:  # Failed to place vertex
            # Fallback: place at regular intervals
            angle = 2 * np.pi * i / num_vertices
            radius = 0.8
            x = 2.0 + radius * np.cos(angle)
            y = 1.5 + radius * np.sin(angle)
            pos.append([x, y])
    
    # Create a simple triangulation (Delaunay-like, but simplified)
    # For simplicity, we'll create triangles from a center point
    center_idx = 0  # Use first vertex as center
    triangles = []
    
    # Create triangles connecting center to pairs of adjacent vertices
    # Validate each triangle before adding
    for i in range(1, num_vertices - 1):
        tri = [center_idx, i, i + 1]
        if validate_triangle(pos, tri):
            triangles.append(tri)
    
    # Also connect last vertex back to second vertex
    if num_vertices > 2:
        tri = [center_idx, num_vertices - 1, 1]
        if validate_triangle(pos, tri):
            triangles.append(tri)
    
    if len(triangles) == 0:
        # Fallback: create a simple valid triangle mesh
        print("  Warning: Generated triangles were degenerate, using fallback mesh")
        if num_vertices >= 3:
            triangles = [[0, 1, 2]]
        if num_vertices >= 4:
            triangles.append([0, 2, 3])
        if num_vertices >= 5:
            triangles.append([0, 3, 4])
    
    # Generate random muscle connections
    # Connect some vertices to create a muscle network
    muscles = []
    num_muscles = min(num_vertices + 2, num_vertices * 2)
    
    # Always connect center to other vertices (but ensure minimum length)
    for i in range(1, num_vertices):
        p1, p2 = np.array(pos[0]), np.array(pos[i])
        dist = np.linalg.norm(p2 - p1)
        if dist > 0.2 and random.random() > 0.3:  # 70% chance, minimum length
            muscles.append([0, i])
    
    # Add some connections between non-center vertices
    for _ in range(num_muscles - len(muscles)):
        v1 = random.randint(1, num_vertices - 1)
        v2 = random.randint(1, num_vertices - 1)
        if v1 != v2:
            p1, p2 = np.array(pos[v1]), np.array(pos[v2])
            dist = np.linalg.norm(p2 - p1)
            if dist > 0.2:  # Minimum muscle length
                if [v1, v2] not in muscles and [v2, v1] not in muscles:
                    muscles.append([v1, v2])
    
    print(f"  Created {len(pos)} vertices, {len(triangles)} triangles, {len(muscles)} muscles")
    
    return pos, triangles, muscles


def visualize_robot(pos, triangles, muscles, num_steps=200, output_file='random_robot.mp4', fps=30):
    """Visualize a robot simulation."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create system
    system = System(device=device)
    system.set(
        pos=pos,
        triangles=triangles,
        muscles=muscles
    )
    
    # Store trajectory
    trajectory = []
    
    print(f"Running simulation for {num_steps} steps...")
    for step in range(num_steps):
        # Random periodic muscle control
        t = step * 0.05
        a = torch.ones(system.num_muscles, device=device)
        for i in range(system.num_muscles):
            # Create varied wave patterns for each muscle
            # Keep muscle actions in reasonable range to prevent collapse
            phase = i * 0.7
            freq = 0.3 + i * 0.1
            a[i] = 0.5 + 0.4 * np.sin(t * freq + phase)  # Range: 0.1 to 0.9
        
        system.a = a
        
        # Run simulation step with more iterations for stability
        system.step(max_optim_iters=100)
        
        # Check for degenerate triangles (safety check)
        pos_np = system.pos0.cpu().numpy()
        triangles_np = np.array(triangles)
        for tri in triangles_np:
            a, b, c = pos_np[tri[0]], pos_np[tri[1]], pos_np[tri[2]]
            ab = b - a
            ac = c - a
            area = abs(ab[0] * ac[1] - ab[1] * ac[0])
            if area < 1e-6:
                print(f"  Warning: Degenerate triangle detected at step {step}, area={area:.2e}")
        
        trajectory.append(pos_np.copy())
        
        if (step + 1) % 50 == 0:
            print(f"  Completed {step + 1}/{num_steps} steps")
    
    # Create visualization
    print("Creating visualization...")
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Set up plot
    all_pos = np.array(trajectory)
    x_min, x_max = all_pos[:, :, 0].min() - 0.5, all_pos[:, :, 0].max() + 0.5
    y_min, y_max = all_pos[:, :, 1].min() - 0.5, all_pos[:, :, 1].max() + 0.5
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Random Robot Simulation')
    ax.grid(True, alpha=0.3)
    
    # Draw ground
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=2, label='Ground')
    
    # Convert to numpy
    triangles = np.array(triangles)
    muscles = np.array(muscles)
    
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
        line, = ax.plot([], [], 'r-', linewidth=2, alpha=0.7)
        muscle_lines.append(line)
    
    vertex_points, = ax.plot([], [], 'ko', markersize=5)
    
    def animate(frame):
        """Animation function."""
        pos_frame = trajectory[frame]
        
        # Update triangles
        for i, tri_idx in enumerate(triangles):
            tri_verts = pos_frame[tri_idx]
            triangle_patches[i].set_xy(tri_verts)
        
        # Update muscles
        for i, muscle_idx in enumerate(muscles):
            p1 = pos_frame[muscle_idx[0]]
            p2 = pos_frame[muscle_idx[1]]
            muscle_lines[i].set_data([p1[0], p2[0]], [p1[1], p2[1]])
        
        # Update vertices
        vertex_points.set_data(pos_frame[:, 0], pos_frame[:, 1])
        
        ax.set_title(f'Random Robot Simulation - Step {frame}/{num_steps-1}')
        
        return triangle_patches + muscle_lines + [vertex_points]
    
    # Create animation
    print("Rendering animation...")
    anim = animation.FuncAnimation(
        fig, animate, frames=num_steps,
        interval=1000/fps, blit=True, repeat=True
    )
    
    # Save video
    print(f"Saving video to {output_file}...")
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps, metadata=dict(artist='Algovivo'), bitrate=1800)
    anim.save(output_file, writer=writer)
    
    print(f"Video saved to {output_file}")
    plt.close()
    
    # Also create a static image showing the initial and final states
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, frame_num in enumerate([0, num_steps - 1]):
        ax = axes[idx]
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
        ax.plot(pos_frame[:, 0], pos_frame[:, 1], 'ko', markersize=5)
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.set_title(f'Step {frame_num}')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    
    plt.tight_layout()
    image_file = output_file.replace('.mp4', '_frames.png')
    plt.savefig(image_file)
    print(f"Static frames saved to {image_file}")
    plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate and visualize random robot topologies')
    parser.add_argument('--vertices', type=int, default=5, help='Number of vertices (3-10 recommended)')
    parser.add_argument('--steps', type=int, default=200, help='Number of simulation steps')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--output', '-o', default='random_robot.mp4', help='Output video filename')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second')
    
    args = parser.parse_args()
    
    # Generate random robot
    pos, triangles, muscles = generate_random_simple_robot(
        num_vertices=args.vertices,
        seed=args.seed
    )
    
    # Visualize
    visualize_robot(pos, triangles, muscles, 
                   num_steps=args.steps,
                   output_file=args.output,
                   fps=args.fps)
    
    print("\n" + "="*60)
    print("Random robot visualization complete!")
    print("="*60)

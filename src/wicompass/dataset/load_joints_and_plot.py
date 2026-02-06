import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D proj)
from pathlib import Path
from typing import List, Optional, Tuple, Union

JOINT_NAMES = [
    'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
    'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
    'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist'
]

BONE_CONNECTIONS = [
    (0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8), (6, 9),
    (7, 10), (8, 11), (9, 12), (9, 13), (9, 14), (12, 15), (13, 16), (14, 17),
    (16, 18), (17, 19), (18, 20), (19, 21)
]

# Body part color scheme
BODY_PART_COLORS = {
    'spine': '#1f77b4',      # blue - spine
    'left_arm': '#ff7f0e',   # orange - left arm
    'right_arm': '#2ca02c',  # green - right arm
    'left_leg': '#d62728',   # red - left leg
    'right_leg': '#9467bd',  # purple - right leg
    'head': '#8c564b'        # brown - head
}

# Bone connection body part classification
BONE_PART_MAPPING = {
    # Spine section
    (0, 3): 'spine', (3, 6): 'spine', (6, 9): 'spine', (9, 12): 'spine',
    # Head
    (12, 15): 'head',
    # Left arm
    (9, 13): 'left_arm', (13, 16): 'left_arm', (16, 18): 'left_arm', (18, 20): 'left_arm',
    # Right arm
    (9, 14): 'right_arm', (14, 17): 'right_arm', (17, 19): 'right_arm', (19, 21): 'right_arm',
    # Left leg
    (0, 1): 'left_leg', (1, 4): 'left_leg', (4, 7): 'left_leg', (7, 10): 'left_leg',
    # Right leg
    (0, 2): 'right_leg', (2, 5): 'right_leg', (5, 8): 'right_leg', (8, 11): 'right_leg',
}


def plot_single_pose(joints: np.ndarray, ax, title: str = "Pose", tokens: Optional[np.ndarray] = None):
    """
    Plot a single pose on the specified 3D axis

    Args:
        joints: (num_joints, 3) numpy array or torch tensor
        ax: matplotlib 3D axis
        title: Figure title
        tokens: Token sequence (optional)
    """
    # Ensure joints is a numpy array
    if hasattr(joints, 'numpy'):
        joints = joints.numpy()
    elif hasattr(joints, 'cpu'):
        joints = joints.cpu().numpy()

    joints = np.asarray(joints)
    if joints.ndim != 2 or joints.shape[1] != 3:
        raise ValueError(f"joints shape must be [J,3], got {joints.shape}")

    if joints.shape[0] != len(JOINT_NAMES):
        print(f"Warning: Expected {len(JOINT_NAMES)} joints, got {joints.shape[0]}")

    # Set background style
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('lightgray')
    ax.yaxis.pane.set_edgecolor('lightgray')
    ax.zaxis.pane.set_edgecolor('lightgray')
    ax.xaxis.pane.set_alpha(0.1)
    ax.yaxis.pane.set_alpha(0.1)
    ax.zaxis.pane.set_alpha(0.1)

    # Plot joint points
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2],
               c='black', s=50, alpha=0.9, edgecolors='white', linewidth=1.5)

    # Plot bone connections
    for connection in BONE_CONNECTIONS:
        joint1_idx, joint2_idx = connection
        if joint1_idx >= joints.shape[0] or joint2_idx >= joints.shape[0]:
            continue

        body_part = BONE_PART_MAPPING.get(connection, 'spine')
        color = BODY_PART_COLORS[body_part]

        x_coords = [joints[joint1_idx, 0], joints[joint2_idx, 0]]
        y_coords = [joints[joint1_idx, 1], joints[joint2_idx, 1]]
        z_coords = [joints[joint1_idx, 2], joints[joint2_idx, 2]]

        ax.plot(x_coords, y_coords, z_coords, color=color, linewidth=4.0, alpha=0.9)

    # Set axes
    ax.set_xlabel('X', fontsize=10)
    ax.set_ylabel('Y', fontsize=10)
    ax.set_zlabel('Z', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)

    # Set equal axis aspect ratio
    max_range = np.array([
        joints[:, 0].max() - joints[:, 0].min(),
        joints[:, 1].max() - joints[:, 1].min(),
        joints[:, 2].max() - joints[:, 2].min()
    ]).max() / 2.0

    mid_x = (joints[:, 0].max() + joints[:, 0].min()) * 0.5
    mid_y = (joints[:, 1].max() + joints[:, 1].min()) * 0.5
    mid_z = (joints[:, 2].max() + joints[:, 2].min()) * 0.5

    # Prevent warning when range is 0
    eps = 1e-6 if max_range == 0 else 0.0
    ax.set_xlim(mid_x - max_range - eps, mid_x + max_range + eps)
    ax.set_ylim(mid_y - max_range - eps, mid_y + max_range + eps)
    ax.set_zlim(mid_z - max_range - eps, mid_z + max_range + eps)

    # Set viewing angle
    ax.view_init(elev=10, azim=0)

    # Add token information display
    if tokens is not None:
        tokens = np.asarray(tokens).reshape(-1)
        tokens_per_line = 16
        token_lines = []
        for i in range(0, len(tokens), tokens_per_line):
            line_tokens = tokens[i:i + tokens_per_line]
            token_lines.append(' '.join(f"{int(t):3d}" for t in line_tokens))

        token_text = "Tokens:\n" + '\n'.join(token_lines)
        ax.text2D(0.02, 0.98, token_text, transform=ax.transAxes,
                  fontsize=6, verticalalignment='top', family='monospace',
                  bbox=dict(boxstyle='round,pad=0.4', facecolor='lightblue', alpha=0.8))


def plot_multiple_poses(poses_data: List, titles: Optional[List[str]] = None,
                        n_cols: int = 4, figsize_per_plot: Tuple[int, int] = (4, 4)):
    """
    Plot overview of multiple poses

    Args:
        poses_data: List of (joints, tokens) tuples or List of joints
        titles: List of titles (optional)
        n_cols: Number of columns per row
        figsize_per_plot: Size of each subplot

    Returns:
        matplotlib figure
    """
    n_poses = len(poses_data)
    n_rows = (n_poses + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(figsize_per_plot[0] * n_cols, figsize_per_plot[1] * n_rows))

    for i, pose_data in enumerate(poses_data):
        ax = fig.add_subplot(n_rows, n_cols, i + 1, projection='3d')

        # Handle different input formats
        if isinstance(pose_data, tuple):
            joints, tokens = pose_data
        else:
            joints, tokens = pose_data, None

        title = titles[i] if titles and i < len(titles) else f"Pose {i + 1}"
        plot_single_pose(joints, ax, title, tokens)

    plt.tight_layout()
    return fig


def ensure_batch_format(arr: np.ndarray) -> np.ndarray:
    """
    Support [22,3] or [N,22,3], normalize to [N,22,3]
    """
    arr = np.asarray(arr)
    if arr.ndim == 2 and arr.shape == (len(JOINT_NAMES), 3):
        return arr[None, ...]
    if arr.ndim == 3 and arr.shape[1:] == (len(JOINT_NAMES), 3):
        return arr
    raise ValueError(f"Unsupported array shape {arr.shape}. Expected [22,3] or [N,22,3].")


def save_figure(fig: plt.Figure, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"[Saved] {out_path}")


def main():
    # Default input/output
    in_path = Path("joint.npy")
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        print(f"Error: {in_path} not found.")
        sys.exit(1)

    data_batch = np.load(in_path)  # Expected [N, 22, 3] or [22, 3]
    data_batch = ensure_batch_format(data_batch)  # -> [N,22,3]
    N = data_batch.shape[0]
    print(f"Loaded {in_path} with shape {data_batch.shape}")

    # Organize plotting data
    poses_data = []
    for i in range(N):
        joints = data_batch[i]
        poses_data.append((joints, None))

    # 1) Plot overview and save
    grid_fig = plot_multiple_poses(poses_data, n_cols=min(4, max(1, N)))
    save_figure(grid_fig, out_dir / "poses_grid.png")
    plt.close(grid_fig)

    # 2) Save each sample separately
    for i, pose in enumerate(poses_data, start=1):
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection='3d')
        if isinstance(pose, tuple):
            joints, tokens = pose
        else:
            joints, tokens = pose, None
        plot_single_pose(joints, ax, title=f"Pose {i}", tokens=tokens)
        save_figure(fig, out_dir / f"pose_{i:03d}.png")
        plt.close(fig)

    print(f"All figures saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()

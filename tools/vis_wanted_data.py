#!/usr/bin/env python3
"""
Interactive Pose Visualization Tool

Navigate through sampled poses using keyboard controls.
Uses visualization functions from src/wicompass/visualization.

Controls:
    Right Arrow / N  : Next pose
    Left Arrow  / P  : Previous pose
    Up Arrow    / J  : Jump forward 10 poses
    Down Arrow  / K  : Jump backward 10 poses
    Home        / G  : Go to first pose
    End         / E  : Go to last pose
    R           : Random pose
    Q / Escape  : Quit
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from wicompass.visualization import plot_single_pose


class InteractivePoseViewer:
    """Interactive viewer for navigating through poses with keyboard controls."""

    def __init__(self, poses: np.ndarray, title: str = "Pose Viewer"):
        """
        Initialize the interactive pose viewer.

        Args:
            poses: Array of shape (N, num_joints, 3) containing poses
            title: Window title
        """
        self.poses = poses
        self.n_poses = len(poses)
        self.current_idx = 0
        self.title = title

        # Create figure
        self.fig = plt.figure(figsize=(12, 10))
        self.fig.canvas.manager.set_window_title(title)

        # Create main axis for pose
        self.ax = self.fig.add_subplot(111, projection="3d")

        # Add slider for pose navigation
        slider_ax = self.fig.add_axes([0.2, 0.02, 0.6, 0.03])
        self.slider = Slider(
            slider_ax,
            "Pose",
            0,
            self.n_poses - 1,
            valinit=0,
            valstep=1,
            valfmt="%d",
        )
        self.slider.on_changed(self._on_slider_change)

        # Connect keyboard events
        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)

        # Initial plot
        self._update_plot()

        # Print instructions
        self._print_instructions()

    def _print_instructions(self):
        """Print keyboard control instructions."""
        print("\n" + "=" * 60)
        print("Interactive Pose Viewer")
        print("=" * 60)
        print(f"Total poses: {self.n_poses}")
        print(f"Pose shape: {self.poses.shape[1:]} (joints, xyz)")
        print("-" * 60)
        print("Keyboard Controls:")
        print("  Right Arrow / N  : Next pose")
        print("  Left Arrow  / P  : Previous pose")
        print("  Up Arrow    / J  : Jump forward 10 poses")
        print("  Down Arrow  / K  : Jump backward 10 poses")
        print("  Home        / G  : Go to first pose")
        print("  End         / E  : Go to last pose")
        print("  R                : Random pose")
        print("  Q / Escape       : Quit")
        print("=" * 60 + "\n")

    def _on_key_press(self, event):
        """Handle keyboard events for navigation."""
        if event.key in ["right", "n"]:
            self._goto_pose(self.current_idx + 1)
        elif event.key in ["left", "p"]:
            self._goto_pose(self.current_idx - 1)
        elif event.key in ["up", "j"]:
            self._goto_pose(self.current_idx + 10)
        elif event.key in ["down", "k"]:
            self._goto_pose(self.current_idx - 10)
        elif event.key in ["home", "g"]:
            self._goto_pose(0)
        elif event.key in ["end", "e"]:
            self._goto_pose(self.n_poses - 1)
        elif event.key == "r":
            self._goto_pose(np.random.randint(0, self.n_poses))
        elif event.key in ["q", "escape"]:
            plt.close(self.fig)
            print("Viewer closed.")

    def _on_slider_change(self, val):
        """Handle slider value change."""
        new_idx = int(val)
        if new_idx != self.current_idx:
            self.current_idx = new_idx
            self._update_plot()

    def _goto_pose(self, idx: int):
        """Navigate to a specific pose index with bounds checking."""
        # Wrap around or clamp
        new_idx = idx % self.n_poses  # Wrap around
        if new_idx != self.current_idx:
            # Update slider first (this will NOT trigger _update_plot because current_idx hasn't changed yet)
            # So we need to update current_idx and call _update_plot manually
            self.current_idx = new_idx
            self.slider.set_val(new_idx)  # Update slider display
            self._update_plot()  # Actually update the plot
        else:
            # Force update if same index (for refresh)
            self._update_plot()

    def _update_plot(self):
        """Update the plot with the current pose."""
        self.ax.clear()

        pose = self.poses[self.current_idx]
        title = f"Pose {self.current_idx + 1} / {self.n_poses}"

        plot_single_pose(
            pose, 
            ax=self.ax, 
            title=title, 
            show_axes=False,
            joint_size=200,  # Larger joints
            line_width=9.0   # Thicker lines
        )

        # Zoom in by adjusting camera distance (default is usually 10)
        # Smaller value = closer camera = larger pose
        self.ax.dist = 5

        self.fig.canvas.draw_idle()

    def show(self):
        """Display the interactive viewer."""
        plt.show()


def load_poses(file_path: str) -> np.ndarray:
    """
    Load poses from a .npy file.

    Args:
        file_path: Path to the .npy file

    Returns:
        Array of poses with shape (N, num_joints, 3)
    """
    poses = np.load(file_path)

    print(f"Loaded poses from: {file_path}")
    print(f"Shape: {poses.shape}")

    # Validate shape
    if poses.ndim != 3:
        raise ValueError(
            f"Expected 3D array (N, joints, 3), got shape {poses.shape}"
        )

    if poses.shape[2] != 3:
        raise ValueError(
            f"Expected last dimension to be 3 (xyz), got {poses.shape[2]}"
        )

    return poses


def main():
    parser = argparse.ArgumentParser(
        description="Interactive visualization of sampled poses",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "pose_file",
        nargs="?",
        default="logs/wicompass/sampled_poses/pps_sampling_k8_quantile9/converted_poses.npy",
        help="Path to the .npy file containing poses (default: %(default)s)",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Starting pose index (default: 0)",
    )

    args = parser.parse_args()

    # Resolve path relative to repo root
    pose_path = Path(args.pose_file)
    if not pose_path.is_absolute():
        repo_root = Path(__file__).resolve().parent.parent
        pose_path = repo_root / pose_path

    if not pose_path.exists():
        print(f"Error: File not found: {pose_path}")
        sys.exit(1)

    # Load poses
    poses = load_poses(str(pose_path))

    # Create and show viewer
    viewer = InteractivePoseViewer(
        poses,
        title=f"Pose Viewer - {pose_path.name}",
    )

    # Set starting pose if specified
    if args.start > 0:
        viewer._goto_pose(args.start)

    viewer.show()


if __name__ == "__main__":
    main()


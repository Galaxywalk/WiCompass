"""
Millimeter-wave Radar Point Cloud + Human Pose Visualizer

This tool visualizes both mmWave point cloud and human pose skeleton for alignment checking.

NPZ file format (pose_pointcloud.npz):
    frame_ids: (N,) int32 - frame indices
    poses: (N, 22, 3) float32 - 22 joints per frame
    pointclouds: (N,) object - variable-length point cloud arrays
    metadata: JSON string with offset info

Controls:
    Space: Play/Pause
    Left/Right or A/D: Previous/Next frame (when paused)
    Up/Down: Increase/Decrease playback speed
    1/3/5: Set point cloud aggregation window
    [ / ]: Adjust point cloud sync offset (] if PC lags behind pose)
    0: Reset sync offset to 0
    R: Reset view to default
    P: Toggle pose visibility
    C: Toggle point cloud visibility
    L: Toggle loop mode
    Q/Escape: Quit
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import argparse
from pathlib import Path
import json


# SMPL-like 22 joint skeleton connections
# Joint indices: 0-pelvis, 1-left_hip, 2-right_hip, 3-spine1, 4-left_knee, 5-right_knee,
# 6-spine2, 7-left_ankle, 8-right_ankle, 9-spine3, 10-left_foot, 11-right_foot,
# 12-neck, 13-left_collar, 14-right_collar, 15-head, 16-left_shoulder, 17-right_shoulder,
# 18-left_elbow, 19-right_elbow, 20-left_wrist, 21-right_wrist
SKELETON_CONNECTIONS = [
    # Spine
    (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
    # Left leg
    (0, 1), (1, 4), (4, 7), (7, 10),
    # Right leg
    (0, 2), (2, 5), (5, 8), (8, 11),
    # Left arm
    (9, 13), (13, 16), (16, 18), (18, 20),
    # Right arm
    (9, 14), (14, 17), (17, 19), (19, 21),
]


class PosePointCloudVisualizer:
    def __init__(self, npz_path: str, fps: int = 10):
        """Initialize the visualizer with a NPZ file path."""
        print(f"Loading data from: {npz_path}")
        data = np.load(npz_path, allow_pickle=True)
        
        # Load data
        self.frame_ids = data['frame_ids']
        self.poses = data['poses']  # (N, 22, 3)
        self.pointclouds = data['pointclouds']  # (N,) object array
        
        # Parse metadata
        if 'metadata' in data:
            self.metadata = json.loads(str(data['metadata'][0]))
        else:
            self.metadata = {}
        
        self.num_frames = len(self.frame_ids)
        total_points = sum(len(pc) for pc in self.pointclouds)
        print(f"Loaded {self.num_frames} frames, {total_points} total points")
        print(f"Pose shape: {self.poses.shape}")
        if self.metadata:
            print(f"Metadata: {self.metadata}")
        
        # Current state
        self.current_frame_idx = 0
        self.aggregation_window = 1  # 1, 3, or 5 frames for point cloud
        self.pointcloud_frame_offset = 0  # Offset for point cloud sync (+ = point cloud ahead, - = behind)
        
        # Playback settings
        self.fps = fps
        self.playing = True
        self.loop = True
        
        # Visibility toggles
        self.show_pose = True
        self.show_pointcloud = True
        
        # Compute global ranges for consistent scaling
        all_points = np.vstack([pc for pc in self.pointclouds if len(pc) > 0])
        all_poses = self.poses.reshape(-1, 3)
        all_data = np.vstack([all_points, all_poses])
        
        self.x_min, self.x_max = all_data[:, 0].min(), all_data[:, 0].max()
        self.y_min, self.y_max = all_data[:, 1].min(), all_data[:, 1].max()
        self.z_min, self.z_max = all_data[:, 2].min(), all_data[:, 2].max()
        
        print(f"X range: [{self.x_min:.3f}, {self.x_max:.3f}]")
        print(f"Y range: [{self.y_min:.3f}, {self.y_max:.3f}]")
        print(f"Z range: [{self.z_min:.3f}, {self.z_max:.3f}]")
        
        # Default view angles (X-Y facing screen, Z as depth)
        # elev=0: horizontal view, azim=-90: looking along Z axis
        self.default_elev = -80
        self.default_azim = -90
        self.current_elev = self.default_elev
        self.current_azim = self.default_azim
        
        # Setup figure
        self.setup_figure()
        
    def setup_figure(self):
        """Setup matplotlib figure and 3D axes."""
        plt.style.use('dark_background')
        
        self.fig = plt.figure(figsize=(14, 10))
        self.ax = self.fig.add_axes([0.05, 0.1, 0.85, 0.85], projection='3d')
        self.fig.canvas.manager.set_window_title('Pose + Point Cloud Visualizer')
        
        # Set initial view
        self.ax.view_init(elev=self.default_elev, azim=self.default_azim)
        
        # Connect keyboard handler
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # Help text
        help_text = 'Space: Play/Pause | ←/→: Frame | ↑/↓: Speed | 1/3/5: Agg | [/]: Sync offset | 0: Reset offset | P/C: Toggle | Q: Quit'
        self.help_text = self.fig.text(0.5, 0.02, help_text, ha='center', fontsize=9, color='gray')
        
        # Title
        self.title_text = self.fig.text(0.5, 0.95, '', ha='center', fontsize=12, color='white')
        
        # Store plot references
        self.scatter = None
        self.skeleton_lines = []
        self.joint_scatter = None
        
        # Initial plot
        self._setup_axes()
        
    def _setup_axes(self):
        """Setup axis labels and limits."""
        self.ax.set_xlabel('X [m] (Horizontal)', fontsize=11, labelpad=10)
        self.ax.set_ylabel('Y [m] (Vertical)', fontsize=11, labelpad=10)
        self.ax.set_zlabel('Z [m] (Depth)', fontsize=11, labelpad=10)
        
        # Fixed axis limits based on actual data ranges
        # Pose: X[-0.6, 0.8], Y[-0.8, 1.1], Z[2.3, 3.0]
        # Use symmetric range around human body
        self.ax.set_xlim(-1.5, 1.5)   # X: horizontal (left-right)
        self.ax.set_ylim(-1.5, 1.5)   # Y: vertical (up-down)
        self.ax.set_zlim(1.5, 4.0)    # Z: depth (distance from radar)
        
    def filter_pointcloud_by_pelvis(self, pointcloud, pelvis_pos):
        """
        Filter point cloud to keep only points within range of pelvis.
        
        Args:
            pointcloud: (N, 3) array of points
            pelvis_pos: (3,) array of pelvis xyz position
            
        Returns:
            Filtered point cloud within:
            - X: ±1.5m from pelvis (3m total)
            - Y: ±1.5m from pelvis (3m total)
            - Z: ±0.5m from pelvis (1m total)
        """
        if len(pointcloud) == 0:
            return pointcloud
        
        # Filter ranges centered on pelvis
        x_range = 1.5  # ±1.5m in X (horizontal)
        y_range = 1.5  # ±1.5m in Y (vertical)
        z_range = 0.5  # ±0.5m in Z (depth)
        
        px, py, pz = pelvis_pos
        
        # Create mask for points within range
        mask = (
            (np.abs(pointcloud[:, 0] - px) <= x_range) &
            (np.abs(pointcloud[:, 1] - py) <= y_range) &
            (np.abs(pointcloud[:, 2] - pz) <= z_range)
        )
        
        return pointcloud[mask]
    
    def get_aggregated_pointcloud(self):
        """Get point cloud from current and adjacent frames based on aggregation window.
        
        The pointcloud_frame_offset allows compensating for synchronization issues:
        - Positive offset: point cloud is fetched from future frames (use if PC appears to lag)
        - Negative offset: point cloud is fetched from past frames (use if PC appears to lead)
        """
        half_window = (self.aggregation_window - 1) // 2
        
        # Apply frame offset for synchronization
        center_idx = self.current_frame_idx + self.pointcloud_frame_offset
        center_idx = max(0, min(self.num_frames - 1, center_idx))
        
        start_idx = max(0, center_idx - half_window)
        end_idx = min(self.num_frames - 1, center_idx + half_window)
        
        # Get current pelvis position (joint 0) for filtering
        pelvis_pos = self.poses[self.current_frame_idx, 0]  # (3,)
        
        # Collect and filter points
        points_list = []
        for i in range(start_idx, end_idx + 1):
            pc = self.pointclouds[i]
            if len(pc) > 0:
                # Filter points around pelvis
                filtered_pc = self.filter_pointcloud_by_pelvis(pc, pelvis_pos)
                if len(filtered_pc) > 0:
                    points_list.append(filtered_pc)
        
        if points_list:
            return np.vstack(points_list)
        else:
            return np.array([]).reshape(0, 3)
    
    def animate(self, frame_num):
        """Animation update function."""
        if self.playing:
            self.current_frame_idx += 1
            if self.current_frame_idx >= self.num_frames:
                if self.loop:
                    self.current_frame_idx = 0
                else:
                    self.current_frame_idx = self.num_frames - 1
                    self.playing = False
        
        return self.update_plot()
        
    def update_plot(self):
        """Update the 3D visualization."""
        # Save view angles
        self.current_elev = self.ax.elev
        self.current_azim = self.ax.azim
        
        # Clear axes
        self.ax.clear()
        
        # Get current data
        pose = self.poses[self.current_frame_idx]  # (22, 3)
        pointcloud = self.get_aggregated_pointcloud()
        
        artists = []
        
        # Draw point cloud
        if self.show_pointcloud and len(pointcloud) > 0:
            self.scatter = self.ax.scatter(
                pointcloud[:, 0], pointcloud[:, 1], pointcloud[:, 2],
                c=pointcloud[:, 2],  # Color by Z (depth)
                cmap='viridis',
                s=10,
                alpha=0.6,
                vmin=1.5, vmax=4.0,
                label=f'Point Cloud ({len(pointcloud)} pts)'
            )
            artists.append(self.scatter)
        
        # Draw skeleton
        if self.show_pose:
            # Draw bones
            for i, j in SKELETON_CONNECTIONS:
                if i < len(pose) and j < len(pose):
                    xs = [pose[i, 0], pose[j, 0]]
                    ys = [pose[i, 1], pose[j, 1]]
                    zs = [pose[i, 2], pose[j, 2]]
                    line, = self.ax.plot(xs, ys, zs, 'r-', linewidth=2, alpha=0.9)
                    artists.append(line)
            
            # Draw joints
            self.joint_scatter = self.ax.scatter(
                pose[:, 0], pose[:, 1], pose[:, 2],
                c='red', s=30, alpha=1.0, marker='o',
                label='Pose Joints'
            )
            artists.append(self.joint_scatter)
        
        # Re-setup axes
        self._setup_axes()
        
        # Restore view
        self.ax.view_init(elev=self.current_elev, azim=self.current_azim)
        
        # Add legend
        if self.show_pose or self.show_pointcloud:
            self.ax.legend(loc='upper right', fontsize=9)
        
        # Update title
        frame_id = self.frame_ids[self.current_frame_idx]
        status = "▶ Playing" if self.playing else "⏸ Paused"
        loop_status = "🔁" if self.loop else ""
        pose_status = "👤" if self.show_pose else ""
        cloud_status = "☁️" if self.show_pointcloud else ""
        
        pc_count = len(pointcloud) if self.show_pointcloud else 0
        offset_str = f"Sync: {self.pointcloud_frame_offset:+d}" if self.pointcloud_frame_offset != 0 else ""
        title = f'Frame {frame_id} ({self.current_frame_idx + 1}/{self.num_frames})'
        title += f'  |  Agg: {self.aggregation_window}  |  Points: {pc_count}'
        if offset_str:
            title += f'  |  {offset_str}'
        title += f'  |  {self.fps} FPS  |  {status} {loop_status} {pose_status} {cloud_status}'
        self.title_text.set_text(title)
        
        return artists
        
    def on_key_press(self, event):
        """Handle keyboard events."""
        if event.key == ' ':
            self.playing = not self.playing
            print(f"Playback: {'Playing' if self.playing else 'Paused'}")
            if not self.playing:
                self.update_plot()
                self.fig.canvas.draw_idle()
            
        elif event.key in ['right', 'd', 'D']:
            self.playing = False
            self.current_frame_idx = min(self.current_frame_idx + 1, self.num_frames - 1)
            self.update_plot()
            self.fig.canvas.draw_idle()
            
        elif event.key in ['left', 'a', 'A']:
            self.playing = False
            self.current_frame_idx = max(self.current_frame_idx - 1, 0)
            self.update_plot()
            self.fig.canvas.draw_idle()
            
        elif event.key == 'up':
            self.fps = min(self.fps + 5, 60)
            print(f"FPS: {self.fps}")
            self.restart_animation()
            
        elif event.key == 'down':
            self.fps = max(self.fps - 5, 1)
            print(f"FPS: {self.fps}")
            self.restart_animation()
            
        elif event.key == '1':
            self.aggregation_window = 1
            print("Aggregation: Single frame")
            
        elif event.key == '3':
            self.aggregation_window = 3
            print("Aggregation: 3 frames (±1)")
            
        elif event.key == '5':
            self.aggregation_window = 5
            print("Aggregation: 5 frames (±2)")
            
        elif event.key in ['r', 'R']:
            self.current_elev = self.default_elev
            self.current_azim = self.default_azim
            self.ax.view_init(elev=self.default_elev, azim=self.default_azim)
            print("View reset")
            self.fig.canvas.draw_idle()
            
        elif event.key in ['p', 'P']:
            self.show_pose = not self.show_pose
            print(f"Pose: {'ON' if self.show_pose else 'OFF'}")
            self.update_plot()
            self.fig.canvas.draw_idle()
            
        elif event.key in ['c', 'C']:
            self.show_pointcloud = not self.show_pointcloud
            print(f"Point Cloud: {'ON' if self.show_pointcloud else 'OFF'}")
            self.update_plot()
            self.fig.canvas.draw_idle()
            
        elif event.key in ['l', 'L']:
            self.loop = not self.loop
            print(f"Loop: {'ON' if self.loop else 'OFF'}")
            
        elif event.key == '[':
            # Point cloud appears to lead pose, shift it backward
            self.pointcloud_frame_offset -= 1
            print(f"Point cloud offset: {self.pointcloud_frame_offset} frames (negative = PC shifted backward)")
            self.update_plot()
            self.fig.canvas.draw_idle()
            
        elif event.key == ']':
            # Point cloud appears to lag pose, shift it forward
            self.pointcloud_frame_offset += 1
            print(f"Point cloud offset: {self.pointcloud_frame_offset} frames (positive = PC shifted forward)")
            self.update_plot()
            self.fig.canvas.draw_idle()
            
        elif event.key == '0':
            # Reset offset
            self.pointcloud_frame_offset = 0
            print("Point cloud offset reset to 0")
            self.update_plot()
            self.fig.canvas.draw_idle()
            
        elif event.key in ['q', 'Q', 'escape']:
            if hasattr(self, 'anim') and self.anim.event_source:
                self.anim.event_source.stop()
            plt.close(self.fig)
            print("Closed.")
    
    def restart_animation(self):
        """Restart animation with new FPS."""
        if hasattr(self, 'anim') and self.anim.event_source:
            self.anim.event_source.stop()
        interval = 1000 // self.fps
        self.anim = FuncAnimation(
            self.fig, 
            self.animate,
            interval=interval,
            blit=False,
            cache_frame_data=False
        )
        self.fig.canvas.draw_idle()
            
    def show(self):
        """Display the visualization."""
        print("\n" + "="*70)
        print("Pose + Point Cloud Visualizer")
        print("="*70)
        print("Controls:")
        print("  Space        : Play/Pause")
        print("  Left/Right   : Previous/Next frame")
        print("  Up/Down      : Increase/Decrease FPS")
        print("  1/3/5        : Point cloud aggregation window")
        print("  [ / ]        : Adjust point cloud sync offset (if PC lags, press ])")
        print("  0            : Reset sync offset to 0")
        print("  P            : Toggle pose visibility")
        print("  C            : Toggle point cloud visibility")
        print("  R            : Reset view")
        print("  L            : Toggle loop")
        print("  Q/Escape     : Quit")
        print("  Mouse drag   : Rotate view")
        print("="*70 + "\n")
        
        # Create animation
        interval = 1000 // self.fps
        self.anim = FuncAnimation(
            self.fig, 
            self.animate,
            interval=interval,
            blit=False,
            cache_frame_data=False
        )
        
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize mmWave point cloud and human pose together',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        'npz_file',
        type=str,
        nargs='?',
        default='pose_pointcloud.npz',
        help='Path to the NPZ file containing pose and point cloud data'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=10,
        help='Playback frames per second (default: 10)'
    )
    parser.add_argument(
        '--start-frame',
        type=int,
        default=0,
        help='Starting frame index (default: 0)'
    )
    parser.add_argument(
        '--aggregation',
        type=int,
        choices=[1, 3, 5],
        default=3,
        help='Initial point cloud aggregation window (default: 3)'
    )
    parser.add_argument(
        '--no-loop',
        action='store_true',
        help='Disable loop playback'
    )
    parser.add_argument(
        '--no-pose',
        action='store_true',
        help='Hide pose skeleton initially'
    )
    parser.add_argument(
        '--no-pointcloud',
        action='store_true',
        help='Hide point cloud initially'
    )
    parser.add_argument(
        '--sync-offset',
        type=int,
        default=0,
        help='Point cloud frame offset for sync (positive: PC ahead, negative: PC behind)'
    )
    
    args = parser.parse_args()
    
    # Handle relative path
    npz_path = Path(args.npz_file)
    if not npz_path.is_absolute():
        script_dir = Path(__file__).parent.parent
        potential_path = script_dir / args.npz_file
        if potential_path.exists():
            npz_path = potential_path
        elif not npz_path.exists():
            print(f"Error: Could not find NPZ file: {args.npz_file}")
            print(f"Tried: {npz_path} and {potential_path}")
            return
    
    if not npz_path.exists():
        print(f"Error: NPZ file not found: {npz_path}")
        return
    
    # Create visualizer
    visualizer = PosePointCloudVisualizer(str(npz_path), fps=args.fps)
    
    # Set initial state
    if args.start_frame > 0:
        visualizer.current_frame_idx = min(args.start_frame, visualizer.num_frames - 1)
    visualizer.aggregation_window = args.aggregation
    visualizer.loop = not args.no_loop
    visualizer.show_pose = not args.no_pose
    visualizer.show_pointcloud = not args.no_pointcloud
    visualizer.pointcloud_frame_offset = args.sync_offset
    
    visualizer.show()


if __name__ == '__main__':
    main()


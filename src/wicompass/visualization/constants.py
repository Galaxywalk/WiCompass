# -*- coding: utf-8 -*-
"""
Visualization Constants

Shared constants for human pose visualization including joint names,
skeleton connections, and color schemes.
"""

# SMPL-style skeleton joint names (22 joints)
JOINT_NAMES = [
    'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
    'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
    'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist'
]

# Skeleton bone connections (parent, child) pairs
BONE_CONNECTIONS = [
    (0, 1), (1, 4), (4, 7), (7, 10),   # Left leg: pelvis -> hip -> knee -> ankle -> foot
    (0, 2), (2, 5), (5, 8), (8, 11),   # Right leg: pelvis -> hip -> knee -> ankle -> foot
    (0, 3), (3, 6), (6, 9),            # Spine: pelvis -> spine1 -> spine2 -> spine3
    (9, 12), (12, 15),                 # Neck/Head: spine3 -> neck -> head
    (9, 13), (13, 16), (16, 18), (18, 20),  # Left arm: spine3 -> collar -> shoulder -> elbow -> wrist
    (9, 14), (14, 17), (17, 19), (19, 21),  # Right arm: spine3 -> collar -> shoulder -> elbow -> wrist
]

# Body part color scheme for visualization
BODY_PART_COLORS = {
    'spine': '#1f77b4',      # Blue
    'left_arm': '#ff7f0e',   # Orange
    'right_arm': '#2ca02c',  # Green
    'left_leg': '#d62728',   # Red
    'right_leg': '#9467bd',  # Purple
    'head': '#8c564b'        # Brown
}

# Mapping from bone connection to body part
BONE_PART_MAPPING = {
    # Spine
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

# Default visualization parameters
DEFAULT_JOINT_SIZE = 50
DEFAULT_LINE_WIDTH = 4.0
DEFAULT_ELEV = 10
DEFAULT_AZIM = 0

# Consolidated default plot settings dictionary
DEFAULT_PLOT_SETTINGS = {
    'joint_size': DEFAULT_JOINT_SIZE,
    'line_width': DEFAULT_LINE_WIDTH,
    'elev': DEFAULT_ELEV,
    'azim': DEFAULT_AZIM,
    'joint_color': '#2c3e50',
    'joint_alpha': 0.9,
    'bone_alpha': 0.8,
    'figsize_single': (10, 8),
    'figsize_per_subplot': (5, 4),
    'dpi': 300,
}

import os 
import numpy as np
from .base_dataset import BaseRadarPoseDataset, cropping


class SimulationMMBody(BaseRadarPoseDataset):
    """
    Simulation Dataset for radar-based pose estimation.
    
    Expected directory structure: {root}/sequence_*/radar/, {root}/sequence_*/label/
    
    Config:
        dataset_path: Path to training data (e.g., rfgen_data/k10_q85)
        validation_dataset_path: Path to validation data (e.g., rfgen_data/benchmark)
    """
    
    def __init__(self, config, split, device=None):
        # Handle separate validation path
        if split != "train" and "validation_dataset_path" in config:
            config = config.copy()
            config["dataset_path"] = config["validation_dataset_path"]
        super().__init__(config, split, device)
        
        # Cache frame ranges for merging
        self._frame_range_cache = {}
    
    def _load_data_list(self):
        """Load file paths from sequence_*/radar/ and sequence_*/label/ structure."""
        # Try with split subdirectory first, then root directly
        split_path = os.path.join(self.root, self.split)
        if not os.path.exists(split_path):
            split_path = self.root
        
        # Find all sequences
        sequences = sorted(
            [d for d in os.listdir(split_path) if d.startswith("sequence_")],
            key=lambda x: int(x.split("_")[-1])
        )
        
        data_list = []
        for seq_name in sequences:
            seq_path = os.path.join(split_path, seq_name)
            radar_dir = os.path.join(seq_path, "radar")
            label_dir = os.path.join(seq_path, "label")
            
            # Get frame files
            radar_files = {int(f.split("_")[1].split(".")[0]): os.path.join(radar_dir, f) 
                          for f in os.listdir(radar_dir) if f.endswith(".npy")}
            label_files = {int(f.split("_")[1].split(".")[0]): os.path.join(label_dir, f)
                          for f in os.listdir(label_dir) if f.endswith(".npy")}
            
            # Match frames that have both radar and label
            common_frames = sorted(set(radar_files.keys()) & set(label_files.keys()))
            for frame in common_frames:
                data_list.append({
                    "frame": frame,
                    "radar": radar_files[frame],
                    "label": label_files[frame],
                    "radar_dir": radar_dir,
                })
            
            print(f"Sequence {seq_name}: {len(common_frames)} samples")
        
        # Apply data quantity filter
        if self.used_data_quantity < 1.0:
            n = int(len(data_list) * self.used_data_quantity)
            indices = np.random.choice(len(data_list), n, replace=False)
            data_list = [data_list[i] for i in indices]
            print(f"Using {n} samples ({self.used_data_quantity*100:.0f}%)")
        
        return data_list
    
    def _load_sample(self, idx):
        """Load and process a single sample."""
        item = self.data_list[idx]
        frame = item["frame"]
        radar_dir = item["radar_dir"]
        
        # Cache frame range for this directory
        if radar_dir not in self._frame_range_cache:
            frames = [int(f.split('_')[1].split('.')[0]) 
                     for f in os.listdir(radar_dir) if f.endswith('.npy')]
            self._frame_range_cache[radar_dir] = (min(frames), max(frames))
        
        min_frame, max_frame = self._frame_range_cache[radar_dir]
        half = self.merge_nframes // 2
        start = max(min_frame, frame - half)
        end = min(max_frame, frame + half) + 1
        
        # Load label
        gt_joints = np.load(item["label"]).astype(np.float32)
        
        # Load merged radar frames
        radar_list = []
        for f in range(start, end):
            path = os.path.join(radar_dir, f"frame_{f}.npy")
            if os.path.exists(path):
                data = np.load(path)[:, :3].astype(np.float32)  # Only xyz
                
                # Normalize relative to pelvis
                if self.normalized:
                    data = data - gt_joints[0]
                    data = cropping(data, x_range=[-1, 1], y_range=[-1, 1], z_range=[-1, 1])
                
                radar_list.append(data)
        
        return radar_list, gt_joints

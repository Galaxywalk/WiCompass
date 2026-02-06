# -*- coding: utf-8 -*-
"""MMFi dataset for mmWave radar-based human pose estimation."""

import os
import sys
import numpy as np
from pathlib import Path

# Add src to path for direct execution
_SRC_DIR = Path(__file__).resolve().parent.parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from pose_estimation.dataset.base_dataset import BaseRadarPoseDataset, cropping


class MMFiDataset(BaseRadarPoseDataset):
    """MMFi dataset with scene/subject/action based split."""
    
    SCENE_SUBJECTS = {
        'E01': [f'S{i:02d}' for i in range(1, 11)],
        'E02': [f'S{i:02d}' for i in range(11, 21)],
        'E03': [f'S{i:02d}' for i in range(21, 31)],
        'E04': [f'S{i:02d}' for i in range(31, 41)],
    }
    SUBJECT_SCENE = {s: e for e, subs in SCENE_SUBJECTS.items() for s in subs}
    ALL_ACTIONS = [f'A{i:02d}' for i in range(1, 28)]
    
    def __init__(self, config, split, device=None):
        super().__init__(config, split, device)
    
    def _load_data_list(self):
        """Build list of data samples based on scene/subject/action config."""
        scenes, subjects, actions = self._decode_split()
        
        data_list = []
        for subject in subjects:
            scene = self.SUBJECT_SCENE[subject]
            for action in actions:
                mmwave_dir = os.path.join(self.root, scene, subject, action, 'mmwave')
                gt_path = os.path.join(self.root, scene, subject, action, 'ground_truth.npy')
                
                if not os.path.exists(mmwave_dir):
                    continue
                
                frames = sorted(os.listdir(mmwave_dir))
                n_frames = int(len(frames) * self.used_data_quantity)
                
                for i, frame in enumerate(frames[:n_frames]):
                    data_list.append({
                        'scene': scene,
                        'subject': subject,
                        'action': action,
                        'gt_path': gt_path,
                        'mmwave_path': os.path.join(mmwave_dir, frame),
                        'idx': i
                    })
        return data_list
    
    def _decode_split(self):
        """Parse config to get scenes, subjects, actions for current split."""
        cfg = self.config['train_dataset'] if self.split == 'train' else self.config['val_dataset']
        
        # Parse scenes
        scenes = self._parse_rule(cfg, 'scenes', list(self.SCENE_SUBJECTS.keys()))
        
        # Build subject pool from scenes
        all_subjects = []
        for s in scenes:
            all_subjects.extend(self.SCENE_SUBJECTS[s])
        
        # Parse subjects and actions
        subjects = self._parse_rule(cfg, 'subjects', all_subjects)
        actions = self._parse_rule(cfg, 'actions', self.ALL_ACTIONS)
        
        return scenes, subjects, actions
    
    def _parse_rule(self, cfg, key, default_list):
        """Parse split rule for scenes/subjects/actions."""
        val = cfg.get(key, 'all')
        
        if val == 'all':
            return default_list
        if isinstance(val, list):
            return val
        if isinstance(val, dict) and val.get('split') == 'random':
            pool = val.get('list', default_list)
            ratio = val.get('ratio', 0.5)
            n = int(len(pool) * ratio)
            idx = np.random.permutation(len(pool))
            sel = idx[:n] if self.split == 'train' else idx[-n:]
            return sorted(np.array(pool)[sel].tolist())
        return default_list
    
    def _load_sample(self, idx):
        """Load radar frames and ground truth joints."""
        item = self.data_list[idx]
        
        # Load ground truth
        gt_all = np.load(item['gt_path'])
        gt_joints = gt_all[item['idx']].astype(np.float32)
        
        # Load merged radar frames
        frame_idx = int(item['mmwave_path'].split("/")[-1].split('.')[0][5:])
        start = max(0, frame_idx - self.merge_nframes // 2)
        end = start + self.merge_nframes
        
        radar_list = []
        for f in range(start, end):
            path = item['mmwave_path'].replace(f'frame{frame_idx:03d}', f'frame{f:03d}')
            if os.path.exists(path):
                with open(path, 'rb') as fp:
                    raw = fp.read()
                data = np.frombuffer(raw, dtype=np.float64).reshape(-1, 5).copy()
                data = data.astype(np.float32)
                
                if self.normalized and data.shape[0] > 0:
                    # Center and crop
                    data[:, :3] -= data[:, :3].mean(axis=0)
                    data = cropping(data)
                    if data.shape[0] > 0:
                        data[:, :3] -= data[:, :3].mean(axis=0)
                
                radar_list.append(data)
        
        return radar_list, gt_joints


if __name__ == '__main__':
    import yaml
    # Use path relative to this file
    config_path = Path(__file__).resolve().parent.parent / "configs" / "mmfi.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    dataset = MMFiDataset(config, split='test', device='cuda')
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        radar, joints = dataset[0]
        print(f"Radar shape: {radar.shape}")
        print(f"Joints shape: {joints.shape}")

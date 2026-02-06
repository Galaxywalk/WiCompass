# src/dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from pathlib import Path
from tqdm.auto import tqdm
from abc import ABC, abstractmethod
import warnings
import json

# --- 1. Preprocessing Utils & Registry ---

class PreprocessingUtils:
    """Common preprocessing utilities."""
    
    @staticmethod
    def pelvis_normalization(joints: np.ndarray) -> np.ndarray:
        """Normalize joints by subtracting pelvis position. Supports (T,J,3) or (J,3)."""
        if joints.ndim == 3:
            pelvis = joints[:, 0:1, :]
        elif joints.ndim == 2:
            pelvis = joints[0:1, :]
        else:
            raise ValueError(f"Unsupported joints shape: {joints.shape}")
        return joints - pelvis
    
    @staticmethod
    def coordinate_transform_y_to_z(joints: np.ndarray) -> np.ndarray:
        """
        Transform coordinate system from Y-down to Z-up.
        Converts (X, Y, Z) -> (X, Z, -Y)
        This is used for RealWorld datasets where height is along -Y axis.
        Supports (T,J,3) or (J,3).
        """
        transformed = joints.copy()
        # Swap Y and Z, and negate the new Z (old Y)
        transformed[..., 1], transformed[..., 2] = joints[..., 2].copy(), -joints[..., 1].copy()
        return transformed
    
    @staticmethod
    def flip_y(joints: np.ndarray) -> np.ndarray:
        """
        Flip Y axis to convert from Y-down to Y-up coordinate system.
        Converts (X, Y, Z) -> (X, -Y, Z)
        
        Use this for datasets where head has negative Y (Y points down).
        Target: XY = frontal plane, Z = depth, Y-up.
        Supports (T,J,3) or (J,3).
        """
        transformed = joints.copy()
        transformed[..., 1] = -joints[..., 1]
        return transformed
    
    @staticmethod
    def swap_yz(joints: np.ndarray) -> np.ndarray:
        """
        Swap Y and Z axes to convert from Z-up to Y-up coordinate system.
        Converts (X, Y, Z) -> (X, Z, Y)
        
        Use this for datasets where Z is the vertical axis.
        Target: XY = frontal plane, Z = depth, Y-up.
        Supports (T,J,3) or (J,3).
        """
        transformed = joints.copy()
        transformed[..., 1] = joints[..., 2].copy()
        transformed[..., 2] = joints[..., 1].copy()
        return transformed
    
    @staticmethod
    def z_up_to_y_down(joints: np.ndarray) -> np.ndarray:
        """
        Convert from Z-up to Y-down coordinate system.
        Converts (X, Y, Z) -> (X, -Z, Y)
        
        Use this for MMBody which uses Z-up, to align with AMASS/MMFi which use Y-down.
        After transform: head has negative Y, feet have positive Y.
        Supports (T,J,3) or (J,3).
        """
        transformed = joints.copy()
        new_y = -joints[..., 2].copy()  # -Z becomes new Y
        new_z = joints[..., 1].copy()   # Y becomes new Z
        transformed[..., 1] = new_y
        transformed[..., 2] = new_z
        return transformed
    
    @staticmethod
    def rotate_z_cw90(joints: np.ndarray) -> np.ndarray:
        """
        Rotate 90 degrees clockwise around Z axis (viewed from above).
        Converts (X, Y, Z) -> (Y, -X, Z)
        
        Use this to rotate body from facing +Y to facing +X direction.
        This aligns with MMBody convention where person faces +X.
        Supports (T,J,3) or (J,3).
        """
        transformed = joints.copy()
        new_x = joints[..., 1].copy()   # new_X = old_Y
        new_y = -joints[..., 0].copy()  # new_Y = -old_X
        transformed[..., 0] = new_x
        transformed[..., 1] = new_y
        return transformed


DATASET_REGISTRY = {}

def register_dataset(name):
    """Decorator to register dataset classes."""
    def decorator(cls):
        DATASET_REGISTRY[name] = cls
        return cls
    return decorator


class BaseJointsDataset(Dataset, ABC):
    """Base class for joints datasets with caching support."""
    
    def __init__(self, root_path: str, num_joints: int, device: str, label: int, name: str, 
                 use_cache: bool = True, cache_dir: str = None, **kwargs):
        self.root_path = Path(root_path)
        self.num_joints = num_joints
        self.device = device
        self.label = label
        self.name = name
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir) if cache_dir else self.root_path / ".cache"
        
        self.joints_data = self._load_with_cache()

    @abstractmethod
    def _get_file_paths(self) -> list[Path]:
        """Return list of data file paths."""
        pass

    def _get_cache_path(self) -> Path:
        return self.cache_dir / f"{self.name}_joints{self.num_joints}.pt"
    
    def _load_with_cache(self) -> torch.Tensor:
        """Load data with caching. First run saves cache, subsequent runs load from cache."""
        cache_path = self._get_cache_path()
        
        # Try loading from cache
        if self.use_cache and cache_path.exists():
            print(f"[INFO] Loading {self.name} from cache: {cache_path}")
            try:
                data = torch.load(cache_path, map_location='cpu', weights_only=True).to(self.device)
                print(f"[INFO] Loaded {len(data)} frames from cache.")
                return data
            except Exception as e:
                print(f"[WARN] Cache load failed, reloading: {e}")
        
        # Load from source
        data = self._load_data()
        
        # Save cache
        if self.use_cache and len(data) > 0:
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                torch.save(data.cpu(), cache_path)
                print(f"[INFO] Cached to: {cache_path}")
            except Exception as e:
                print(f"[WARN] Cache save failed: {e}")
        
        return data

    def _load_data(self) -> torch.Tensor:
        """Load data from source files. Override for custom loading logic."""
        file_paths = self._get_file_paths()
        if not file_paths:
            print(f"[WARN] No files found in {self.root_path}")
            return torch.empty(0, self.num_joints, 3, dtype=torch.float32, device=self.device)

        print(f"[INFO] Loading {len(file_paths)} files from {self.root_path.name}...")

        # Scan to count total frames
        total_frames = 0
        for fp in tqdm(file_paths, desc=f"Scanning {self.root_path.name}", leave=False):
            try:
                with np.load(fp) as data:
                    total_frames += len(data['joints'])
            except Exception as e:
                print(f"[WARN] Could not read {fp}: {e}")
        
        if total_frames == 0:
            return torch.empty(0, self.num_joints, 3, dtype=torch.float32, device=self.device)

        # Pre-allocate and load
        joints_data = torch.empty((total_frames, self.num_joints, 3), dtype=torch.float32, device=self.device)
        idx = 0
        for fp in tqdm(file_paths, desc=f"Loading {self.root_path.name}", leave=False):
            try:
                with np.load(fp) as data:
                    joints = self.preprocess(data['joints'].astype(np.float32))
                    n = joints.shape[0]
                    joints_data[idx:idx+n] = torch.from_numpy(joints)
                    idx += n
            except Exception as e:
                print(f"[WARN] Could not load {fp}: {e}")
        
        if idx < total_frames:
            joints_data = joints_data[:idx]
        
        print(f"[INFO] Loaded {len(joints_data)} frames for {self.root_path.name}.")
        return joints_data

    def preprocess(self, joints: np.ndarray) -> np.ndarray:
        """Preprocessing hook. Override for custom preprocessing."""
        return joints

    def __len__(self):
        return self.joints_data.shape[0]

    def __getitem__(self, idx):
        return self.joints_data[idx], self.label


# --- 2. Dataset Implementations ---

@register_dataset('amass')
class AMASSJointsDataset(BaseJointsDataset):
    """AMASS dataset loader with pelvis normalization.
    
    AMASS uses Y-down coordinate system (head has negative Y, feet have positive Y).
    This is the standard coordinate system for Wi-Compass.
    """

    def _get_file_paths(self) -> list[Path]:
        if not self.root_path.exists():
            print(f"[WARN] AMASS dir not found: {self.root_path}")
            return []
        return list(self.root_path.rglob(f"*joints{self.num_joints}.npz"))

    def preprocess(self, joints: np.ndarray) -> np.ndarray:
        return PreprocessingUtils.pelvis_normalization(joints)


@register_dataset('mmbody')
class MMBodyJointsDataset(BaseJointsDataset):
    """MMBody dataset loader (one frame per file).
    
    Original MMBody data uses Z-up coordinate system (head has highest Z).
    Converts to Y-down coordinate system to match other datasets (AMASS, MMFi, Real-world).
    Transform: (X, Y, Z) -> (X, -Z, Y) so that head has negative Y.
    """
    
    def _get_file_paths(self) -> list[Path]:
        if not self.root_path.exists():
            print(f"[WARN] MMBody dir not found: {self.root_path}")
            return []
        files = []
        for split in ['train', 'test']:
            split_dir = self.root_path / split
            if split_dir.exists():
                files.extend(split_dir.rglob("mesh/frame_*.npz"))
        return sorted(files)

    def _load_data(self) -> torch.Tensor:
        file_paths = self._get_file_paths()
        if not file_paths:
            return torch.empty(0, self.num_joints, 3, dtype=torch.float32, device=self.device)

        print(f"[INFO] Loading {len(file_paths)} frames from MMBody...")

        # Validate first file
        with np.load(file_paths[0]) as data:
            if data['joints'].shape[0] < self.num_joints:
                raise ValueError(f"Requested {self.num_joints} joints, file has {data['joints'].shape[0]}")

        joints_data = torch.empty((len(file_paths), self.num_joints, 3), dtype=torch.float32, device=self.device)
        for i, fp in enumerate(tqdm(file_paths, desc="Loading MMBody", leave=False)):
            try:
                with np.load(fp) as data:
                    joints = self.preprocess(data['joints'][:self.num_joints].astype(np.float32))
                    joints_data[i] = torch.from_numpy(joints)
            except Exception as e:
                warnings.warn(f"Failed to load {fp.name}: {e}")
                joints_data[i].fill_(0)
        
        print(f"[INFO] Loaded {len(joints_data)} frames for MMBody.")
        return joints_data

    def preprocess(self, joints: np.ndarray) -> np.ndarray:
        # Keep original Z-up coordinate system
        # Note: AMASS-pretrained VQ-VAE works better with Z-up MMBody data (15mm vs 60mm)
        return PreprocessingUtils.pelvis_normalization(joints)


@register_dataset('wi-compass')
class WiCompassJointsDataset(BaseJointsDataset):
    """Wi-Compass dataset loader with configurable file patterns."""
    
    def __init__(self, root_path: str, num_joints: int, device: str, label: int, name: str, **kwargs):
        self.splits = kwargs.get('splits', ['train', 'val'])
        self.file_pattern = kwargs.get('file_pattern', 'label/frame_*.npy')
        self.subdataset_name = kwargs.get('subdataset_name', name)
        # Coordinate transform: convert Y-down to Z-up coordinate system
        self.coord_transform = kwargs.get('coord_transform', None)
        super().__init__(root_path, num_joints, device, label, name, **kwargs)
    
    def _get_file_paths(self) -> list[Path]:
        if not self.root_path.exists():
            print(f"[WARN] {self.subdataset_name} dir not found: {self.root_path}")
            return []
        files = []
        for split in self.splits:
            split_dir = self.root_path / split
            if split_dir.exists():
                files.extend(split_dir.rglob(self.file_pattern))
            else:
                print(f"[INFO] Split '{split}' not found, skipping...")
        return sorted(files)

    def _load_data(self) -> torch.Tensor:
        file_paths = self._get_file_paths()
        if not file_paths:
            return torch.empty(0, self.num_joints, 3, dtype=torch.float32, device=self.device)

        print(f"[INFO] Loading {len(file_paths)} frames from {self.subdataset_name}...")

        # Validate first file
        data = self._load_single_file(file_paths[0])
        if data.shape[0] < self.num_joints:
            raise ValueError(f"Requested {self.num_joints} joints, file has {data.shape[0]}")

        joints_data = torch.empty((len(file_paths), self.num_joints, 3), dtype=torch.float32, device=self.device)
        for i, fp in enumerate(tqdm(file_paths, desc=f"Loading {self.subdataset_name}", leave=False)):
            try:
                joints = self._load_single_file(fp)[:self.num_joints].astype(np.float32)
                joints_data[i] = torch.from_numpy(self.preprocess(joints))
            except Exception as e:
                warnings.warn(f"Failed to load {fp.name}: {e}")
                joints_data[i].fill_(0)
        
        print(f"[INFO] Loaded {len(joints_data)} frames for {self.subdataset_name}.")
        return joints_data
    
    def _load_single_file(self, file_path: Path) -> np.ndarray:
        """Load single file (.npy or .npz)."""
        if file_path.suffix == '.npy':
            return np.load(file_path)
        elif file_path.suffix == '.npz':
            with np.load(file_path) as data:
                for key in ['joints', 'data', 'pose']:
                    if key in data:
                        return data[key]
                return data[list(data.keys())[0]] if data.keys() else None
        raise ValueError(f"Unsupported format: {file_path.suffix}")

    def preprocess(self, joints: np.ndarray) -> np.ndarray:
        """Preprocess joints with optional coordinate transform and pelvis normalization."""
        # Apply coordinate transform if specified
        if self.coord_transform == 'y_to_z':
            joints = PreprocessingUtils.coordinate_transform_y_to_z(joints)
        elif self.coord_transform == 'flip_y':
            joints = PreprocessingUtils.flip_y(joints)
        elif self.coord_transform == 'swap_yz':
            joints = PreprocessingUtils.swap_yz(joints)
        return PreprocessingUtils.pelvis_normalization(joints)


@register_dataset('mmfi')
class MMFiJointsDataset(BaseJointsDataset):
    """MMFi dataset loader.
    
    Uses mmfi_poses_smplx.npz which contains body_joints in Y-up coordinate system.
    Converts to Z-up to match MMBody and other datasets (head has positive Z).
    Transform: (X, Y, Z) -> (X, Z, Y) (swap Y and Z)
    """
    
    # Fixed file name for MMFi poses
    MMFI_POSES_FILE = "mmfi_poses_smplx.npz"
    
    def __init__(self, root_path: str, num_joints: int, device: str, label: int, name: str, **kwargs):
        super().__init__(root_path, num_joints, device, label, name, **kwargs)
    
    def _get_file_paths(self) -> list[Path]:
        if not self.root_path.exists() or not self.root_path.is_dir():
            print(f"[WARN] MMFi dir not found: {self.root_path}")
            return []
        
        npz_file = self.root_path / self.MMFI_POSES_FILE
        if npz_file.exists():
            return [npz_file]
        
        print(f"[WARN] MMFi file not found: {npz_file}")
        return []

    def _load_data(self) -> torch.Tensor:
        file_paths = self._get_file_paths()
        if not file_paths:
            return torch.empty(0, self.num_joints, 3, dtype=torch.float32, device=self.device)

        data_file = file_paths[0]
        print(f"[INFO] Loading MMFi from {data_file.name}...")

        try:
            data = np.load(data_file)
            if 'body_joints' not in data:
                print(f"[ERROR] 'body_joints' not found in {data_file.name}")
                return torch.empty(0, self.num_joints, 3, dtype=torch.float32, device=self.device)
            
            joints22 = data['body_joints'][:, :self.num_joints, :].astype(np.float32)
            print(f"[INFO] MMFi shape: {joints22.shape}")

            # Apply preprocessing (pelvis norm + coordinate transform)
            processed = self.preprocess(joints22)
            joints_data = torch.from_numpy(processed).to(self.device)
            
        except Exception as e:
            print(f"[ERROR] Failed to load MMFi: {e}")
            return torch.empty(0, self.num_joints, 3, dtype=torch.float32, device=self.device)
        
        print(f"[INFO] Loaded {len(joints_data)} frames for MMFi.")
        return joints_data

    def preprocess(self, joints: np.ndarray) -> np.ndarray:
        # Pelvis normalization first
        joints = PreprocessingUtils.pelvis_normalization(joints)
        # Convert from Y-up to Z-up: (X, Y, Z) -> (X, Z, Y)
        joints = PreprocessingUtils.swap_yz(joints)
        return joints


@register_dataset('real-world')
class RealWorldJointsDataset(BaseJointsDataset):
    """Real-world dataset loader for individual .npy files.
    
    Expected structure:
        root_path/
            label/
                1.npy, 2.npy, ... (each file: (num_joints, 3))
            radar/
                ...
            metadata.json (optional)
    """
    
    def __init__(self, root_path: str, num_joints: int, device: str, label: int, name: str, **kwargs):
        self.coord_transform = kwargs.get('coord_transform', None)
        super().__init__(root_path, num_joints, device, label, name, **kwargs)
    
    def _get_file_paths(self) -> list[Path]:
        label_dir = self.root_path / "label"
        if not label_dir.exists():
            print(f"[WARN] Label dir not found: {label_dir}")
            return []
        
        # Find all .npy files and sort by frame number
        files = list(label_dir.glob("*.npy"))
        # Sort numerically by filename (1.npy, 2.npy, ... 10.npy, 11.npy, ...)
        files = sorted(files, key=lambda f: int(f.stem))
        return files
    
    def _load_data(self) -> torch.Tensor:
        file_paths = self._get_file_paths()
        if not file_paths:
            return torch.empty(0, self.num_joints, 3, dtype=torch.float32, device=self.device)
        
        print(f"[INFO] Loading {len(file_paths)} frames from {self.name}...")
        
        joints_data = torch.empty((len(file_paths), self.num_joints, 3), dtype=torch.float32, device=self.device)
        for i, fp in enumerate(tqdm(file_paths, desc=f"Loading {self.name}", leave=False)):
            try:
                joints = np.load(fp)[:self.num_joints].astype(np.float32)
                joints_data[i] = torch.from_numpy(self.preprocess(joints))
            except Exception as e:
                warnings.warn(f"Failed to load {fp.name}: {e}")
                joints_data[i].fill_(0)
        
        print(f"[INFO] Loaded {len(joints_data)} frames for {self.name}.")
        return joints_data
    
    def preprocess(self, joints: np.ndarray) -> np.ndarray:
        """Preprocess with optional coordinate transform and pelvis normalization."""
        if self.coord_transform == 'y_to_z':
            joints = PreprocessingUtils.coordinate_transform_y_to_z(joints)
        elif self.coord_transform == 'y_to_z_rotate':
            # Y-down to Z-up + rotate to face +X (like MMBody)
            joints = PreprocessingUtils.coordinate_transform_y_to_z(joints)
            joints = PreprocessingUtils.rotate_z_cw90(joints)
        elif self.coord_transform == 'flip_y':
            joints = PreprocessingUtils.flip_y(joints)
        elif self.coord_transform == 'swap_yz':
            joints = PreprocessingUtils.swap_yz(joints)
        elif self.coord_transform == 'rotate_z_cw90':
            joints = PreprocessingUtils.rotate_z_cw90(joints)
        return PreprocessingUtils.pelvis_normalization(joints)


@register_dataset('wicompass')
class WiCompassSimulationDataset(BaseJointsDataset):
    """WiCompass simulation dataset loader for sequence-based .npy files.
    
    Expected structure:
        root_path/
            sequence_0/
                label/
                    frame_1.npy, frame_2.npy, ... (each file: (num_joints, 3))
                radar/
                    ...
            sequence_1/
                label/...
            ...
    """
    
    def __init__(self, root_path: str, num_joints: int, device: str, label: int, name: str, **kwargs):
        self.coord_transform = kwargs.get('coord_transform', None)
        super().__init__(root_path, num_joints, device, label, name, **kwargs)
    
    def _get_file_paths(self) -> list[Path]:
        if not self.root_path.exists():
            print(f"[WARN] Dataset dir not found: {self.root_path}")
            return []
        
        # Find all sequence_* folders
        sequence_dirs = sorted(
            [d for d in self.root_path.iterdir() if d.is_dir() and d.name.startswith('sequence_')],
            key=lambda d: int(d.name.split('_')[1])
        )
        
        if not sequence_dirs:
            print(f"[WARN] No sequence_* folders found in {self.root_path}")
            return []
        
        # Collect all frame files from all sequences
        files = []
        for seq_dir in sequence_dirs:
            label_dir = seq_dir / "label"
            if label_dir.exists():
                seq_files = list(label_dir.glob("frame_*.npy"))
                # Sort by frame number
                seq_files = sorted(seq_files, key=lambda f: int(f.stem.split('_')[1]))
                files.extend(seq_files)
        
        print(f"[INFO] Found {len(files)} frames across {len(sequence_dirs)} sequences")
        return files
    
    def _load_data(self) -> torch.Tensor:
        file_paths = self._get_file_paths()
        if not file_paths:
            return torch.empty(0, self.num_joints, 3, dtype=torch.float32, device=self.device)
        
        print(f"[INFO] Loading {len(file_paths)} frames from {self.name}...")
        
        joints_data = torch.empty((len(file_paths), self.num_joints, 3), dtype=torch.float32, device=self.device)
        for i, fp in enumerate(tqdm(file_paths, desc=f"Loading {self.name}", leave=False)):
            try:
                joints = np.load(fp)[:self.num_joints].astype(np.float32)
                joints_data[i] = torch.from_numpy(self.preprocess(joints))
            except Exception as e:
                warnings.warn(f"Failed to load {fp.name}: {e}")
                joints_data[i].fill_(0)
        
        print(f"[INFO] Loaded {len(joints_data)} frames for {self.name}.")
        return joints_data
    
    def preprocess(self, joints: np.ndarray) -> np.ndarray:
        """Preprocess with optional coordinate transform and pelvis normalization."""
        if self.coord_transform == 'y_to_z':
            joints = PreprocessingUtils.coordinate_transform_y_to_z(joints)
        elif self.coord_transform == 'y_to_z_rotate':
            joints = PreprocessingUtils.coordinate_transform_y_to_z(joints)
            joints = PreprocessingUtils.rotate_z_cw90(joints)
        elif self.coord_transform == 'flip_y':
            joints = PreprocessingUtils.flip_y(joints)
        elif self.coord_transform == 'swap_yz':
            joints = PreprocessingUtils.swap_yz(joints)
        elif self.coord_transform == 'rotate_z_cw90':
            joints = PreprocessingUtils.rotate_z_cw90(joints)
        return PreprocessingUtils.pelvis_normalization(joints)


# --- 3. Config Management ---

def load_dataset_configs(config_path: str = None) -> dict:
    """Load dataset config from JSON file."""
    if config_path is None:
        # Default: src/wicompass/configs/datasets.json
        config_path = Path(__file__).parent.parent / "configs" / "datasets.json"
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"[WARN] Config load failed: {e}")
        return {"amass_datasets": [], "mmbody_datasets": [], "mmfi_datasets": [], "wicompass_datasets": []}


def get_available_datasets(dataset_types: list[str] = None, config_path: str = None) -> list[dict]:
    """Get list of available datasets by type."""
    configs = load_dataset_configs(config_path)
    if dataset_types is None:
        dataset_types = ['amass', 'mmbody', 'mmfi', 'real-world', 'wicompass']
    
    datasets = []
    type_map = {
        'amass': 'amass_datasets',
        'mmbody': 'mmbody_datasets', 
        'mmfi': 'mmfi_datasets',
        'real-world': 'real_world_datasets',
        'wicompass': 'wicompass_datasets'
    }
    for t in dataset_types:
        datasets.extend(configs.get(type_map.get(t, ''), []))
    return datasets


# --- 4. Cache Management ---

def clear_dataset_cache(dataset_path: str = None, cache_dir: str = None):
    """Clear dataset cache."""
    import shutil
    path = Path(cache_dir) if cache_dir else (Path(dataset_path) / ".cache" if dataset_path else None)
    if path and path.exists():
        shutil.rmtree(path)
        print(f"[INFO] Cleared cache: {path}")


# --- 5. Dataset Factory ---

def create_dataset(dataset_configs: list[dict], num_joints: int, device: str,
                   use_cache: bool = True, cache_dir: str = None) -> tuple[ConcatDataset, dict]:
    """Create and concatenate multiple datasets from config."""
    datasets = []
    label_to_name = {}
    
    print("\n--- Creating Datasets ---")
    if use_cache:
        print(f"[INFO] Cache: {cache_dir or 'per-dataset .cache/'}")
    
    for i, config in enumerate(dataset_configs):
        dtype, name, path = config.get('type'), config.get('name'), config.get('path')
        if not all([dtype, name, path]) or dtype not in DATASET_REGISTRY:
            continue
            
        print(f"-> {name} ({dtype})")
        kwargs = config.get('params', {})
        kwargs['use_cache'] = use_cache
        if cache_dir:
            kwargs['cache_dir'] = cache_dir
        
        try:
            ds = DATASET_REGISTRY[dtype](root_path=path, num_joints=num_joints, 
                                         device=device, label=i, name=name, **kwargs)
            if len(ds) > 0:
                datasets.append(ds)
                label_to_name[i] = name
            else:
                print(f"[WARN] {name} is empty, skipping.")
        except Exception as e:
            print(f"[ERROR] {name}: {e}")

    if not datasets:
        raise RuntimeError("No valid datasets loaded.")

    combined = ConcatDataset(datasets)
    print(f"\n--- Summary: {len(combined):,} total samples ---")
    for ds in datasets:
        print(f"  Label {ds.label}: {ds.name} ({len(ds):,})")
    
    return combined, {'label_to_name': label_to_name, 'num_datasets': len(label_to_name)}


# --- 6. Example Usage ---

if __name__ == "__main__":
    NUM_JOINTS = 22
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=== Dataset Test ===")
    try:
        configs = get_available_datasets(['amass'])
        print(f"Found {len(configs)} datasets")
        
        dataset, info = create_dataset(configs, num_joints=NUM_JOINTS, device=DEVICE)
        print(f"Total samples: {len(dataset)}")
        
        # Test DataLoader
        loader = DataLoader(dataset, batch_size=4, shuffle=True,
                           collate_fn=lambda b: (torch.stack([x[0] for x in b]), 
                                                 torch.tensor([x[1] for x in b])))
        joints, labels = next(iter(loader))
        print(f"Batch: {joints.shape}, Labels: {labels.tolist()}")
        print(f"Labels: {[info['label_to_name'][l.item()] for l in labels]}")
        
        # Save for visualization with load_joints_and_plot.py
        np.save("joint.npy", joints.cpu().numpy())
        print("Saved joint.npy for visualization")
        print("✅ Test passed!")

    except Exception as e:
        import traceback
        print(f"❌ Failed: {e}")
        traceback.print_exc()

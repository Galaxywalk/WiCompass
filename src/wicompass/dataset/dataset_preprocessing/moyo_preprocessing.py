#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MOYO Data Preprocessing - Convert pose parameters to joint positions

MOYO data format:
- All SMPLX parameters stored in a single .npz file
- Contains: poses, betas, trans, root_orient, pose_body, pose_hand, pose_jaw, pose_eye
- No separate neutral_stagei.npz file needed

Usage example:
python moyo_preprocessing.py \
  --data_root /path/to/AMASS/MOYO \
  --output_root /path/to/wicompass_workspace/AMASS_preproc \
  --model_root model_zoo \
  --device cuda:0 \
  --frame_stride 2 \
  --chunk_size 512 \
  --fp16
"""

import argparse, os, time, json, math
from pathlib import Path
import numpy as np
import torch
from tqdm.auto import tqdm
from human_body_prior.body_model.body_model import BodyModel

def iter_moyo_files(raw_root: Path):
    """Iterate through MOYO data structure and get all .npz files"""
    # Check if .npz files are directly contained (e.g., /path/to/MOYO/*.npz)
    direct_files = list(raw_root.glob("*.npz"))
    if direct_files:
        print(f"[INFO] Found {len(direct_files)} npz files directly in {raw_root}")
        for mfile in direct_files:
            try:
                npz = np.load(str(mfile))
                if 'poses' not in npz or 'betas' not in npz:
                    continue
                    
                betas = npz.get('betas', np.zeros((1, 16)))
                if betas.ndim == 1:
                    betas = betas.reshape(1, -1)
                if betas.shape[1] > 16:
                    betas = betas[:, :16]
                elif betas.shape[1] < 16:
                    betas = np.pad(betas, ((0, 0), (0, 16 - betas.shape[1])))
                
                yield {
                    "motion_path": mfile,
                    "betas": betas,
                    "npz_data": npz
                }
                
            except Exception as e:
                print(f"[WARN] Failed to load {mfile}: {e}")
                continue
        return
    
    # Iterate through subdirectories to find .npz files
    for sub in raw_root.iterdir():
        if not sub.is_dir(): 
            continue
        
        print(f"[INFO] Checking directory: {sub}")
        
        # Check if .npz files are directly in subdirectory (e.g., MOYO/extra/*.npz)
        sub_files = list(sub.glob("*.npz"))
        if sub_files:
            print(f"[INFO] Found {len(sub_files)} npz files in {sub}")
            for mfile in sub_files:
                try:
                    npz = np.load(str(mfile))
                    if 'poses' not in npz or 'betas' not in npz:
                        continue
                        
                    betas = npz.get('betas', np.zeros((1, 16)))
                    if betas.ndim == 1:
                        betas = betas.reshape(1, -1)
                    if betas.shape[1] > 16:
                        betas = betas[:, :16]
                    elif betas.shape[1] < 16:
                        betas = np.pad(betas, ((0, 0), (0, 16 - betas.shape[1])))
                    
                    yield {
                        "motion_path": mfile,
                        "betas": betas,
                        "npz_data": npz
                    }
                    
                except Exception as e:
                    print(f"[WARN] Failed to load {mfile}: {e}")
                    continue
        else:
            # Continue checking second-level subdirectories (original logic)
            for seq in sub.iterdir():
                if not seq.is_dir(): 
                    continue
                    
                for mfile in seq.glob("*.npz"):
                    try:
                        npz = np.load(str(mfile))
                        if 'poses' not in npz or 'betas' not in npz:
                            continue
                            
                        betas = npz.get('betas', np.zeros((1, 16)))
                        if betas.ndim == 1:
                            betas = betas.reshape(1, -1)
                        if betas.shape[1] > 16:
                            betas = betas[:, :16]
                        elif betas.shape[1] < 16:
                            betas = np.pad(betas, ((0, 0), (0, 16 - betas.shape[1])))
                        
                        yield {
                            "motion_path": mfile,
                            "betas": betas,
                            "npz_data": npz
                        }
                        
                    except Exception as e:
                        print(f"[WARN] Failed to load {mfile}: {e}")
                        continue

def load_moyo_poses(npz):
    """
    Extract poses and trans from MOYO npz file
    MOYO contains complete SMPLX parameters: poses (T, 165), trans (T, 3)
    """
    if 'poses' in npz:
        poses = npz['poses']  # (T, 165) - complete SMPLX poses
    else:
        # If complete poses are not available, try to combine individual parts
        root_orient = npz.get('root_orient', np.zeros((1, 3)))
        pose_body = npz.get('pose_body', np.zeros((1, 63)))
        pose_hand = npz.get('pose_hand', np.zeros((1, 90)))
        pose_jaw = npz.get('pose_jaw', np.zeros((1, 3)))
        pose_eye = npz.get('pose_eye', np.zeros((1, 6)))
        
        # Ensure all parts have the same temporal dimension
        T = max(root_orient.shape[0], pose_body.shape[0])
        if root_orient.shape[0] != T:
            root_orient = np.repeat(root_orient, T, axis=0)
        if pose_body.shape[0] != T:
            pose_body = np.repeat(pose_body, T, axis=0)
        if pose_hand.shape[0] != T:
            pose_hand = np.repeat(pose_hand, T, axis=0)
        if pose_jaw.shape[0] != T:
            pose_jaw = np.repeat(pose_jaw, T, axis=0)
        if pose_eye.shape[0] != T:
            pose_eye = np.repeat(pose_eye, T, axis=0)
            
        # Combine complete poses (T, 165)
        poses = np.concatenate([
            root_orient,  # 3
            pose_body,    # 63
            pose_hand,    # 90
            pose_jaw,     # 3
            pose_eye      # 6
        ], axis=1)
    
    # Extract trans
    if 'trans' in npz:
        trans = npz['trans']
    elif 'transl' in npz:
        trans = npz['transl']
    else:
        trans = np.zeros((poses.shape[0], 3))
        
    # Ensure trans is a 2D array (T, 3)
    if trans.ndim == 1:
        trans = trans.reshape(1, -1)
        
    return poses, trans

def forward_chunked(body_model, poses, trans, betas, device, num_joints, chunk_size):
    """
    Forward pass using SMPLX model
    poses: (T, 165) - complete SMPLX parameters
    Chunked forward pass -> joints (T, num_joints, 3), pelvis-centered
    """
    T = poses.shape[0]
    results = []
    
    with torch.inference_mode():
        for start in range(0, T, chunk_size):
            end = min(start + chunk_size, T)
            sub = poses[start:end]
            sub_trans = trans[start:end]
            sub_betas = betas[start:end]
            
            poses_t = torch.from_numpy(sub).float().to(device, non_blocking=True)
            trans_t = torch.from_numpy(sub_trans).float().to(device, non_blocking=True)
            betas_t = torch.from_numpy(sub_betas[:, :16]).float().to(device, non_blocking=True)

            # Decompose SMPLX parameters
            root_orient = poses_t[:, :3]                    # (T, 3)
            pose_body = poses_t[:, 3:66]                    # (T, 63)
            pose_hand = poses_t[:, 66:156]                  # (T, 90)
            pose_jaw = poses_t[:, 156:159]                  # (T, 3)
            pose_eye = poses_t[:, 159:165]                  # (T, 6)

            body_parms = {
                'root_orient': root_orient,
                'pose_body': pose_body,
                'pose_hand': pose_hand,
                'pose_jaw': pose_jaw,
                'pose_eye': pose_eye,
                'trans': trans_t,
                'betas': betas_t,
            }

            out = body_model(**body_parms)
            joints = out.Jtr[:, :num_joints, :]  # Take first num_joints joints
            # Pelvis centering: subtract position of joint 0 (pelvis)
            joints = joints - joints[:, 0:1, :]
            results.append(joints.cpu())
            
    joints_all = torch.cat(results, dim=0)  # (T, J, 3)
    return joints_all

def save_joints(out_path: Path, joints: torch.Tensor, meta: dict, fp16: bool):
    """Save processed joint data"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = joints.numpy() if not fp16 else joints.half().numpy()
    np.savez_compressed(out_path, joints=arr, meta=json.dumps(meta, ensure_ascii=False))

def preprocess(args):
    raw_root = Path(args.data_root).expanduser().resolve()
    out_root = Path(args.output_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # Use SMPLX model (MOYO uses SMPLX format)
    bm_path = f"{args.model_root}/smplx/{args.gender}/model.npz"
    print(f"[INFO] Loading SMPLX BodyModel: {bm_path}")
    
    try:
        body_model = BodyModel(bm_fname=bm_path, num_betas=args.num_betas).to(args.device)
        body_model.eval()
    except Exception as e:
        print(f"[ERROR] Failed to load body model: {e}")
        print("[INFO] Make sure SMPLX model files are available")
        return

    # Get all MOYO files
    motions = list(iter_moyo_files(raw_root))
    print(f"[INFO] Total MOYO motion files: {len(motions)}")

    if len(motions) == 0:
        print("[ERROR] No valid MOYO files found!")
        return

    # Statistics
    skipped, processed, total_frames_out = 0, 0, 0

    for info in tqdm(motions, desc="Preprocessing MOYO motions", dynamic_ncols=True):
        src = info["motion_path"]
        npz_data = info["npz_data"]
        
        # Output path: mirror original relative path
        rel = src.relative_to(raw_root)
        # e.g. xxx.npz -> xxx_joints22.npz
        stem = rel.stem
        dst = out_root / rel.parent / f"{stem}_joints{args.num_joints}.npz"

        if dst.exists() and not args.overwrite:
            skipped += 1
            continue

        try:
            # Extract poses and trans from already loaded data
            poses, trans = load_moyo_poses(npz_data)
            
            if poses is None:
                print(f"[WARN] No valid poses in {src}, skip.")
                continue

            T = poses.shape[0]
            
            # Apply frame stride
            idx = np.arange(0, T, args.frame_stride)
            if args.max_frames_per_seq is not None:
                idx = idx[:args.max_frames_per_seq]
                
            poses = poses[idx]
            trans = trans[idx]

            # Process betas - betas are fixed in MOYO
            betas = info["betas"]
            if betas.shape[0] == 1:
                # betas are the same for all frames
                betas = np.repeat(betas, poses.shape[0], axis=0)
            elif betas.shape[0] != poses.shape[0]:
                # Adjust betas length to match poses
                if betas.shape[0] > poses.shape[0]:
                    betas = betas[:poses.shape[0]]
                else:
                    betas = np.concatenate([
                        betas, 
                        np.repeat(betas[-1:], poses.shape[0] - betas.shape[0], axis=0)
                    ], axis=0)

            # Forward pass to get joint positions
            joints = forward_chunked(
                body_model,
                poses=poses,
                trans=trans,
                betas=betas,
                device=args.device,
                num_joints=args.num_joints,
                chunk_size=args.chunk_size
            )

            # Prepare metadata
            meta = {
                "source_file": str(src),
                "dataset": "MOYO",
                "num_frames": int(joints.shape[0]),
                "frame_stride": args.frame_stride,
                "centered_pelvis": True,
                "fp16": bool(args.fp16),
                "num_joints": args.num_joints,
                "gender": args.gender,
                "model_type": "SMPLX"
            }
            
            # Save processing results
            save_joints(dst, joints, meta, fp16=args.fp16)
            processed += 1
            total_frames_out += joints.shape[0]

        except Exception as e:
            print(f"[ERROR] Failed to process {src}: {e}")
            continue

    print("========== SUMMARY ==========")
    print(f"Processed files : {processed}")
    print(f"Skipped (exists): {skipped}")
    print(f"Total frames out: {total_frames_out}")
    print(f"Output root     : {out_root}")
    print("=============================")

def parse_args():
    p = argparse.ArgumentParser("MOYO → joints preprocessing")
    p.add_argument("--data_root", type=str, required=True, help="MOYO data root directory")
    p.add_argument("--output_root", type=str, required=True, help="Output root directory (mirror structure)")
    p.add_argument("--model_root", type=str, required=True, help="Contains smplx/<gender>/model.npz")
    p.add_argument("--gender", type=str, default="neutral", choices=["neutral","male","female"], 
                   help="SMPLX model gender to use")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num_joints", type=int, default=22, help="Number of output joints")
    p.add_argument("--num_betas", type=int, default=16, help="Number of shape parameters")
    p.add_argument("--frame_stride", type=int, default=1, help="Frame sampling stride")
    p.add_argument("--max_frames_per_seq", type=int, default=None, help="Maximum frames per sequence")
    p.add_argument("--chunk_size", type=int, default=512, help="Forward pass temporal chunk size")
    p.add_argument("--fp16", action="store_true", help="Save with float16 precision")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    t0 = time.time()
    preprocess(args)
    print(f"[DONE] Total preprocessing time: {time.time() - t0:.2f}s")

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AMASS Data Preprocessing - Convert pose parameters to joint positions

Usage example:
python amass_preprocessing.py \
  --data_root /path/to/AMASS \
  --output_root /path/to/wicompass_workspace/AMASS_preproc \
  --model_root model_zoo \
  --gender neutral \
  --device cuda:0 \
  --frame_stride 1 \
  --chunk_size 512 \
  --fp16
"""

import argparse, os, time, json, math
from pathlib import Path
import numpy as np
import torch
from tqdm.auto import tqdm
from human_body_prior.body_model.body_model import BodyModel

def iter_motion_files(raw_root: Path):
    """Iterate through two-level structure (dataset/sequence/motion.npz), also supports HuMMan format"""
    for sub in raw_root.iterdir():
        if not sub.is_dir(): continue
        
        # Check if it's a HuMMan format directory structure
        if sub.name == "recon_smpl_params":
            # HuMMan format: recon_smpl_params/p000598_a001234/smpl_params/*.npz
            for person_seq in sub.iterdir():
                if not person_seq.is_dir(): continue
                smpl_params_dir = person_seq / "smpl_params"
                if not smpl_params_dir.exists(): continue
                
                # Get betas from the first npz file
                npz_files = list(smpl_params_dir.glob("*.npz"))
                if not npz_files: continue
                
                try:
                    first_npz = np.load(str(npz_files[0]))
                    betas = first_npz.get('betas', np.zeros((1,16)))
                    if betas.ndim == 1:
                        betas = betas.reshape(1, -1)
                    if betas.shape[1] < 16:
                        betas = np.pad(betas, ((0,0),(0,16-betas.shape[1])))
                except Exception:
                    continue
                
                for mfile in npz_files:
                    yield {
                        "motion_path": mfile,
                        "betas": betas
                    }
        else:
            # Original AMASS format: dataset/sequence/neutral_stagei.npz + *.npz
            for seq in sub.iterdir():
                if not seq.is_dir(): continue
                neutral = seq / "neutral_stagei.npz"
                if not neutral.exists(): 
                    continue
                try:
                    static_np = np.load(str(neutral))
                except Exception:
                    continue
                betas = static_np.get('betas', np.zeros((1,16)))
                betas = betas.reshape(1, -1)
                if betas.shape[1] < 16:
                    betas = np.pad(betas, ((0,0),(0,16-betas.shape[1])))
                for mfile in seq.glob("*.npz"):
                    if mfile.name == "neutral_stagei.npz":
                        continue
                    yield {
                        "motion_path": mfile,
                        "betas": betas
                    }

def load_poses(npz):
    """Unified return of poses (T, D), trans (T,3)"""
    if 'poses' in npz:
        poses = npz['poses']
    elif 'pose_body' in npz:
        body = npz['pose_body']
        root = npz.get('root_orient', np.zeros((body.shape[0], 3)))
        poses = np.concatenate([root, body], axis=1)
    elif 'body_pose' in npz and 'global_orient' in npz:
        # HuMMan format: body_pose + global_orient
        body = npz['body_pose']
        root = npz['global_orient']
        
        # Ensure it's a 2D array (T, D)
        if body.ndim == 1:
            body = body.reshape(1, -1)
        if root.ndim == 1:
            root = root.reshape(1, -1)
            
        poses = np.concatenate([root, body], axis=1)
    else:
        return None, None
    
    # Support both 'transl' and 'trans' naming conventions
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
    poses: (T, D)
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

            P = poses_t.shape[1]
            root_orient = poses_t[:, :3]
            pose_body = poses_t[:, 3:66] if P >= 66 else torch.zeros(poses_t.size(0), 63, device=device)
            pose_hand = poses_t[:, 66:156] if P >= 156 else torch.zeros(poses_t.size(0), 90, device=device)
            jaw = poses_t[:, 156:159] if P >= 159 else None
            eye = poses_t[:, 159:165] if P >= 165 else None

            body_parms = {
                'root_orient': root_orient,
                'pose_body': pose_body,
                'pose_hand': pose_hand,
                'trans': trans_t,
                'betas': betas_t,
            }
            if jaw is not None: body_parms['pose_jaw'] = jaw
            if eye is not None: body_parms['pose_eye'] = eye

            out = body_model(**body_parms)
            joints = out.Jtr[:, :num_joints, :]
            joints = joints - joints[:, 0:1, :]
            results.append(joints.cpu())
            
    joints_all = torch.cat(results, dim=0)  # (T, J, 3)
    return joints_all

def save_joints(out_path: Path, joints: torch.Tensor, meta: dict, fp16: bool):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = joints.numpy() if not fp16 else joints.half().numpy()
    np.savez_compressed(out_path, joints=arr, meta=json.dumps(meta, ensure_ascii=False))

def preprocess(args):
    raw_root = Path(args.data_root).expanduser().resolve()
    out_root = Path(args.output_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    bm_path = f"{args.model_root}/smplx/{args.gender}/model.npz"
    print(f"[INFO] Loading BodyModel: {bm_path}")
    body_model = BodyModel(bm_fname=bm_path, num_betas=args.num_betas).to(args.device)
    body_model.eval()

    motions = list(iter_motion_files(raw_root))
    print(f"[INFO] Total motion files: {len(motions)}")

    # Statistics
    skipped, processed, total_frames_out = 0, 0, 0

    for info in tqdm(motions, desc="Preprocess motions", dynamic_ncols=True):
        src = info["motion_path"]
        # Output path: mirror original relative path
        rel = src.relative_to(raw_root)
        # e.g. xxx.npz -> xxx_joints22.npz
        stem = rel.stem
        dst = out_root / rel.parent / f"{stem}_joints{args.num_joints}.npz"

        if dst.exists() and not args.overwrite:
            skipped += 1
            continue

        try:
            npz = np.load(str(src))
        except Exception as e:
            print(f"[WARN] Skip (load fail): {src} | {e}")
            continue

        poses, trans = load_poses(npz)
        if poses is None:
            print(f"[WARN] No pose keys in {src}, skip.")
            continue

        T = poses.shape[0]
        # stride
        idx = np.arange(0, T, args.frame_stride)
        if args.max_frames_per_seq is not None:
            idx = idx[:args.max_frames_per_seq]
        poses = poses[idx]
        trans = trans[idx]

        betas = info["betas"]
        if betas.shape[0] == 1:
            betas = np.repeat(betas, poses.shape[0], axis=0)
        elif betas.shape[0] != poses.shape[0]:
            # Safety: truncate/pad
            if betas.shape[0] > poses.shape[0]:
                betas = betas[:poses.shape[0]]
            else:
                betas = np.concatenate([betas, np.repeat(betas[-1:], poses.shape[0]-betas.shape[0], axis=0)], axis=0)

        joints = forward_chunked(
            body_model,
            poses=poses,
            trans=trans,
            betas=betas,
            device=args.device,
            num_joints=args.num_joints,
            chunk_size=args.chunk_size
        )

        meta = {
            "source_file": str(src),
            "num_frames": int(joints.shape[0]),
            "frame_stride": args.frame_stride,
            "centered_pelvis": True,
            "fp16": bool(args.fp16),
            "num_joints": args.num_joints,
            "gender": args.gender
        }
        save_joints(dst, joints, meta, fp16=args.fp16)
        processed += 1
        total_frames_out += joints.shape[0]

    print("========== SUMMARY ==========")
    print(f"Processed files : {processed}")
    print(f"Skipped (exists): {skipped}")
    print(f"Total frames out: {total_frames_out}")
    print(f"Output root     : {out_root}")
    print("=============================")

def parse_args():
    p = argparse.ArgumentParser("AMASS → joints (first 22) preprocessing")
    p.add_argument("--data_root", type=str, required=True, help="Original AMASS root directory")
    p.add_argument("--output_root", type=str, required=True, help="Output root directory (mirror structure)")
    p.add_argument("--model_root", type=str, required=True, help="Contains smplx/<gender>/model.npz")
    p.add_argument("--gender", type=str, default="neutral", choices=["neutral","male","female"])
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num_joints", type=int, default=22)
    p.add_argument("--num_betas", type=int, default=16)
    p.add_argument("--frame_stride", type=int, default=1)
    p.add_argument("--max_frames_per_seq", type=int, default=None)
    p.add_argument("--chunk_size", type=int, default=512, help="Temporal chunk size")
    p.add_argument("--fp16", action="store_true", help="Save as float16")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    t0 = time.time()
    preprocess(args)
    print(f"[DONE] Total preprocessing time: {time.time() - t0:.2f}s")
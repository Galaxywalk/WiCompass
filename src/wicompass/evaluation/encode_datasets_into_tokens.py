import torch
import numpy as np
import argparse
from pathlib import Path
from wicompass.evaluation.core import load_config, load_model, create_dataloader
from wicompass.evaluation.encoder import ModelEncoder, save_tokens
from wicompass.dataset import create_dataset, get_available_datasets


def parse_args():
    parser = argparse.ArgumentParser(description="Encode pose datasets into tokens using VQVAE model")
    parser.add_argument("--config", type=str, 
                        default="src/wicompass/configs/joint_vae_base_tokennum16_tokenclass64.json",
                        help="Path to model config file")
    parser.add_argument("--model", type=str,
                        default="logs/vqvae/vqvae_tokennum16_tokenclass64/best_model.pth",
                        help="Path to model checkpoint")
    parser.add_argument("--output-dir", type=str,
                        default="logs/wicompass/encoded_tokens",
                        help="Output directory for encoded tokens")
    parser.add_argument("--dataset", type=str,
                        default="RealWorld-1225",
                        help="Dataset name to encode")
    parser.add_argument("--dataset-type", type=str,
                        default="wicompass",
                        choices=["amass", "mmbody", "mmfi", "real-world", "wicompass"],
                        help="Dataset type category")
    parser.add_argument("--batch-size", type=int,
                        default=128,
                        help="Batch size for encoding")
    parser.add_argument("--device", type=str,
                        default="cuda",
                        help="Device to use (cuda or cpu)")
    return parser.parse_args()


# ============ Main Pipeline ============
if __name__ == "__main__":
    args = parse_args()
    
    print(f"🔧 Configuration:")
    print(f"   Config: {args.config}")
    print(f"   Model:  {args.model}")
    print(f"   Output: {args.output_dir}")
    print(f"   Dataset: {args.dataset} ({args.dataset_type})")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Device: {args.device}")
    print()
    
    # 1. Load config and model
    config = load_config(args.config)
    model = load_model(config['model'], args.model, device=args.device)

    # 2. Create dataset
    dataset_configs = [c for c in get_available_datasets([args.dataset_type]) 
                       if c['name'] == args.dataset]
    
    if not dataset_configs:
        raise ValueError(f"Dataset '{args.dataset}' not found in type '{args.dataset_type}'!")
    
    dataset, info = create_dataset(dataset_configs, num_joints=22, device=args.device)
    print(f"📊 Dataset: {args.dataset}, samples: {len(dataset)}")

    # 3. Create DataLoader
    dataloader = create_dataloader(dataset, batch_size=args.batch_size, shuffle=False)

    # 4. Encode
    encoder = ModelEncoder(model, device=args.device)
    results = encoder.encode_dataset(dataloader)

    # 5. Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as HDF5 format
    output_h5 = output_dir / f"{args.dataset}_tokens.h5"
    tokens = results['tokens'].astype(np.int8)
    metadata = {
        'total_samples': results['total_samples'],
        'num_tokens_per_sample': tokens.shape[1] if len(tokens) > 0 else 0,
        'codebook_size': config['model'].get('token_class_num', 64),
        'dataset_name': args.dataset,
    }
    save_tokens(tokens, results['labels'], str(output_h5), metadata)
    print(f"✅ Saved HDF5: {output_h5}")

    
    print(f"\n📈 Results summary:")
    print(f"   - Tokens shape: {tokens.shape}")
    print(f"   - Token range: [{tokens.min()}, {tokens.max()}]")
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VQ-VAE Dataset Encoding Script
Encode pose dataset to discrete token sequences

Examples:
    python -m wicompass.evaluation.scripts.encode_dataset \
        --model logs/vqvae/best_model.pth \
        --config configs/joint_vae_base.json \
        --output logs/encoded_tokens.h5
"""

import argparse
from pathlib import Path
import sys

# Add src/ to Python path for absolute imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from wicompass.evaluation.encoder import encode_dataset
from wicompass.evaluation.core import load_config


def generate_output_filename(config_path: str, base_output: str) -> str:
    """
    Generate output filename based on datasets in config file.
    
    Args:
        config_path: Path to config file
        base_output: Base output path
        
    Returns:
        Modified output path with dataset names
    """
    try:
        config = load_config(config_path)
        datasets = config.get('data', {}).get('datasets', [])
        
        if datasets:
            dataset_str = "-".join(sorted(datasets))
            output_path = Path(base_output)
            stem = output_path.stem
            suffix = output_path.suffix
            parent = output_path.parent
            
            if stem == "encoded_tokens":
                new_filename = f"{dataset_str}_tokens{suffix}"
            else:
                new_filename = f"{stem}_{dataset_str}{suffix}"
            
            return str(parent / new_filename)
        return base_output
    except Exception as e:
        print(f"⚠️ Error generating filename, using default: {e}")
        return base_output


def main():
    parser = argparse.ArgumentParser(
        description='VQ-VAE Dataset Encoding Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--model', required=True, help='Model weights path')
    parser.add_argument('--config', required=True, help='Config file path')
    parser.add_argument('--output', default='logs/wicompass/encoded_tokens.h5',
                       help='Output file path (dataset info added automatically)')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--device', default='cuda', help='Compute device')
    
    args = parser.parse_args()
    
    # Generate output filename with dataset info
    final_output_path = generate_output_filename(args.config, args.output)
    
    print("🔄 VQ-VAE Dataset Encoding")
    print("=" * 60)
    print(f"📁 Config: {args.config}")
    print(f"📁 Output: {final_output_path}")
    
    try:
        # Run encoding
        print("🔄 Starting dataset encoding...")
        results = encode_dataset(
            model_path=args.model,
            config_path=args.config,
            output_path=final_output_path,
            batch_size=args.batch_size,
            device=args.device
        )
        
        # Print summary
        print("\n" + "=" * 60)
        print("📋 Encoding Results Summary")
        print("=" * 60)
        print(f"🎯 Total samples: {results['total_samples']:,}")
        print(f"🎯 Token range: {results['token_range'][0]} - {results['token_range'][1]}")
        print(f"💾 File size: {results['file_size_mb']:.2f} MB")
        print(f"📁 Output file: {results['output_path']}")
        print("=" * 60)
        print("✅ Encoding complete!")
        return 0
        
    except Exception as e:
        print(f"❌ Error during encoding: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())


#!/usr/bin/env python3
import numpy as np
from pathlib import Path
import argparse
import sys




def main():
    parser = argparse.ArgumentParser(
        description="Build F matrix from multiple .npy files, each treated as a column.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=
        """
            Examples:
                --files file1.npy file2.npy file3.npy --out F.npy
                --files /path/to/9dof-e*.npy --out F.npy
        """
    )
    parser.add_argument(
        "--files",
        nargs="+",
        required=True,
        help="Paths to .npy files, each treated as a column and stacked to form the F matrix in the order provided."
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output path for the F matrix .npy file."
    )
    
    args = parser.parse_args()
    
    # Convert to Path objects
    files = [Path(f) for f in args.files]
    
    if not files:
        print("Error: No files provided.", file=sys.stderr)
        sys.exit(1)
    
    print(f"Loading {len(files)} files...")
    for i, f in enumerate(files, 1):
        if not f.exists():
            print(f"Error: File {f} does not exist.", file=sys.stderr)
            sys.exit(1)
        print(f"  {i}. {f}")
    
    # Load columns
    cols = [np.load(f) for f in files]
    
    # Stack as columns
    F = np.column_stack(cols)
    print(f"\nF matrix shape: {F.shape}")
    
    # Save to output path
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, F)
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
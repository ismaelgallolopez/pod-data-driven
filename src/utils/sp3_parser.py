import os
import numpy as np
from datetime import datetime
import torch

def parse_sp3_line(line):
    """Parses a Position line from an SP3 file."""
    try:
        # Fixed-width format: P SCID  X(km)         Y(km)         Z(km)
        x = float(line[4:18].strip())
        y = float(line[18:32].strip())
        z = float(line[32:46].strip())
        return x, y, z
    except (ValueError, IndexError):
        return None, None, None

def process_sp3_file(file_path, sc_id=None, origin_dt=None):
    """Extracts time and ECEF coordinates from an SP3 file.

    Args:
        file_path: path to .sp3 file
        sc_id: optional satellite id string (e.g., 'L06' or 'G01'). If provided,
               only lines starting with f'P{sc_id}' are parsed. If None, all 'P' lines are
               considered (backwards compatible).
        origin_dt: optional datetime to use as the global time origin. If None,
                   the first epoch in the first file processed becomes the origin.

    Returns:
        epochs (np.array), positions (np.array), origin_dt_used (datetime)
    """
    epochs = []
    positions = []
    origin = origin_dt

    with open(file_path, 'r') as f:
        current_t = None
        for line in f:
            if line.startswith('/*'):  # EOF
                break

            if line.startswith('* '):  # Epoch header line
                parts = line.split()
                try:
                    year = int(parts[1])
                    month = int(parts[2])
                    day = int(parts[3])
                    hour = int(parts[4])
                    minute = int(parts[5])
                    second = int(float(parts[6]))

                    dt = datetime(year, month, day, hour, minute, second)
                    if origin is None:
                        origin = dt
                    current_t = (dt - origin).total_seconds()
                except (ValueError, IndexError):
                    continue

            else:
                # Satellite position lines start with 'P<satid>' like 'PL06' or 'PG01'
                if current_t is None:
                    continue
                if sc_id is not None:
                    prefix = f'P{sc_id}'
                    if not line.startswith(prefix):
                        continue
                else:
                    if not line.startswith('P'):
                        continue

                x, y, z = parse_sp3_line(line)
                if x is not None:
                    epochs.append(current_t)
                    positions.append([x, y, z])

    return np.array(epochs), np.array(positions), origin

def run_extraction(input_dir='inputs', output_dir='data/processed', sc_id=None):
    os.makedirs(output_dir, exist_ok=True)
    
    noisy_data = []
    truth_data = []

    files = [f for f in os.listdir(input_dir) if f.endswith('.sp3')]
    if not files:
        print(f"No .sp3 files found in {input_dir}")
        return

    origin_dt = None
    for file in sorted(files):
        path = os.path.join(input_dir, file)
        print(f"Processing {file}...")
        t, pos, origin_dt = process_sp3_file(path, sc_id=sc_id, origin_dt=origin_dt)

        if len(t) == 0:
            continue

        # Stack as [t, x, y, z]
        combined = np.column_stack((t, pos))
        
        if 'SPPLEO' in file:
            noisy_data.append(combined)
        elif 'ODCP' in file: # Use ODCP as Ground Truth
            truth_data.append(combined)

    # Save as PyTorch Tensors
    if noisy_data:
        noisy_tensor = torch.tensor(np.vstack(noisy_data), dtype=torch.float32)
        torch.save(noisy_tensor, os.path.join(output_dir, 'noisy_spp.pt'))
        print(f"Total Noisy samples: {noisy_tensor.shape}")
        
    if truth_data:
        truth_tensor = torch.tensor(np.vstack(truth_data), dtype=torch.float32)
        torch.save(truth_tensor, os.path.join(output_dir, 'truth_odcp.pt'))
        print(f"Total Truth samples: {truth_tensor.shape}")
    
    print(f"\nFiles saved to {output_dir}")

if __name__ == "__main__":
    run_extraction()
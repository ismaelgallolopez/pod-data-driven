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

def process_sp3_file(file_path):
    """Extracts time and ECEF coordinates from an SP3 file."""
    epochs = []
    positions = []
    start_dt = None

    with open(file_path, 'r') as f:
        current_t = None
        for line in f:
            if line.startswith('/*'): # EOF
                break
                
            if line.startswith('* '):  # Epoch header line
                parts = line.split()
                try:
                    # Indexing into parts after split: 0='*', 1=YYYY, 2=MM, 3=DD, 4=HH, 5=MM, 6=SS(.sss)
                    year = int(parts[1])
                    month = int(parts[2])
                    day = int(parts[3])
                    hour = int(parts[4])
                    minute = int(parts[5])
                    second = int(float(parts[6]))

                    dt = datetime(year, month, day, hour, minute, second)
                    if start_dt is None:
                        start_dt = dt
                    current_t = (dt - start_dt).total_seconds()
                except (ValueError, IndexError):
                    continue
            
            elif line.startswith('P') and current_t is not None:
                x, y, z = parse_sp3_line(line)
                if x is not None:
                    epochs.append(current_t)
                    positions.append([x, y, z])

    return np.array(epochs), np.array(positions)

def run_extraction(input_dir='inputs', output_dir='data/processed'):
    os.makedirs(output_dir, exist_ok=True)
    
    noisy_data = []
    truth_data = []

    files = [f for f in os.listdir(input_dir) if f.endswith('.sp3')]
    if not files:
        print(f"No .sp3 files found in {input_dir}")
        return

    for file in sorted(files):
        path = os.path.join(input_dir, file)
        print(f"Processing {file}...")
        t, pos = process_sp3_file(path)
        
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
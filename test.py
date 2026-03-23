import os
os.environ.setdefault('TORCH_COMPILE_DISABLE', '1')

import torch
from src.training.train_pinn import train_pinn

def run_quick_check(epochs=10, subset=1000):
    data_path = 'data/processed/noisy_spp.pt'
    if not os.path.exists(data_path):
        print("Data not found. Run: python src/utils/sp3_parser.py first.")
        return

    data = torch.load(data_path)
    t = data[:, 0]
    r = data[:, 1:4]

    num_samples = len(t)
    print(f"Total Samples: {num_samples}")

    # Subsample for quick test
    subset = int(os.environ.get('TEST_DATASET_SIZE', str(subset)))
    if num_samples > subset:
        torch.manual_seed(0)
        perm = torch.randperm(num_samples)[:subset]
        t_small = t[perm]
        r_small = r[perm]
        print(f"Using reduced training set: {subset} / {num_samples} samples")
    else:
        t_small = t
        r_small = r

    # Run a short training (no resume by default to be deterministic for quick tests)
    model = train_pinn(t_small, r_small, epochs=epochs, batch_size=512,
                       resume=False, checkpoint_dir='data/processed/test_ckpt', save_freq=5)
    print("Quick-check finished. Checkpoints and model saved to data/processed/test_ckpt")


def choose_devices():
    devices = []
    try:
        import torch_directml
        devices.append((torch_directml.device(), 'DirectML'))
    except Exception:
        pass
    if torch.cuda.is_available():
        devices.append((torch.device('cuda'), 'CUDA'))
    devices.append((torch.device('cpu'), 'CPU'))
    return devices


def sync_device(device):
    if getattr(device, 'type', None) == 'cuda':
        torch.cuda.synchronize()
    else:
        torch.tensor(0, device=device).cpu()


def benchmark_model(device, batch_size=256, iters=30, warmup=10):
    from src.models.pinn import KinematicPINN
    import time, statistics

    model = KinematicPINN().to(device)
    model.train()

    batch_t = torch.randn(batch_size, 1, device=device, dtype=torch.float32)
    batch_r = torch.randn(batch_size, 3, device=device, dtype=torch.float32)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def run_iters(do_backward):
        for _ in range(warmup):
            out = model(batch_t)
            if do_backward:
                loss = ((out - batch_r)**2).mean()
                loss.backward()
                optimizer.zero_grad()
                optimizer.step()
        sync_device(device)

        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            out = model(batch_t)
            if do_backward:
                loss = ((out - batch_r)**2).mean()
                loss.backward()
                optimizer.zero_grad()
                optimizer.step()
            sync_device(device)
            times.append(time.perf_counter() - t0)
        return statistics.mean(times), statistics.stdev(times)

    fwd_mean, fwd_std = run_iters(do_backward=False)
    train_mean, train_std = run_iters(do_backward=True)

    return {
        'device': str(device), 'name': getattr(device, 'name', str(device)), 'batch': batch_size,
        'fwd_mean': fwd_mean, 'fwd_std': fwd_std,
        'train_mean': train_mean, 'train_std': train_std
    }


def print_summary(results):
    print('\nBenchmark Summary:')
    hdr = f"{'Device':<12} {'Batch':>6} {'Fwd(ms)':>10} {'Fwd std':>10} {'Fwd s/s':>10} {'Train(ms)':>10} {'Train std':>10} {'Train s/s':>10}"
    print(hdr)
    print('-' * len(hdr))
    for r in results:
        fwd_ms = r['fwd_mean'] * 1000
        fwd_std = r['fwd_std'] * 1000
        fwd_ss = r['batch'] / r['fwd_mean'] if r['fwd_mean'] > 0 else 0
        train_ms = r['train_mean'] * 1000
        train_std = r['train_std'] * 1000
        train_ss = r['batch'] / r['train_mean'] if r['train_mean'] > 0 else 0
        name = r.get('name', r['device'])
        print(f"{name:<12} {r['batch']:6d} {fwd_ms:10.3f} {fwd_std:10.3f} {fwd_ss:10.1f} {train_ms:10.3f} {train_std:10.3f} {train_ss:10.1f}")

if __name__ == '__main__':
    run_quick_check()
    # Benchmark on available devices and print summary
    results = []
    for dev, name in choose_devices():
        try:
            print(f"\nRunning benchmark on {name} -> {dev}")
            res = benchmark_model(dev, batch_size=512, iters=20, warmup=5)
            results.append(res)
        except Exception as e:
            print(f"Skipping {name}: {e}")
    if results:
        print_summary(results)

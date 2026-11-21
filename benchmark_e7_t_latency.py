"""Benchmark one E7-T checkpoint on CUDA and write results for the report notebook."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pandas as pd
import torch

from pneunet_e7_t import load_checkpoint
from report_e7_t import write_json
from train_pneunet_e7_t import ROOT, configure_runtime, load_p3, sha256


def other_gpu_processes(uuid):
    """Sample external GPU utilization before and after each timing session."""
    if shutil.which('nvidia-smi') is None:
        return None
    try:
        result = subprocess.run(['nvidia-smi', 'pmon', '-i', uuid, '-c', '1', '-s', 'u'],
                                capture_output=True, text=True, check=True, timeout=10)
        rows = [line.split() for line in result.stdout.splitlines() if not line.startswith('#')]
        return [int(row[1]) for row in rows if len(row) >= 5 and row[1].isdigit()
                and int(row[1]) != os.getpid() and any(value.isdigit() and int(value) > 0 for value in row[3:5])]
    except (OSError, subprocess.SubprocessError):
        return None


@torch.inference_mode()
def measure(model, images, device, warmup, repeats):
    for _ in range(warmup):
        model(images)
    start = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    stop = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    # Initialize CUDA events before the measured work.
    for first, last in zip(start, stop):
        first.record()
        last.record()
    torch.cuda.synchronize(device)
    for first, last in zip(start, stop):
        first.record()
        model(images)
        last.record()
    torch.cuda.synchronize(device)
    return [first.elapsed_time(last) for first, last in zip(start, stop)]


def summarize(trials):
    records = []
    for (model, batch), group in trials.groupby(['model', 'batch'], sort=False):
        times = group.ms_batch.to_numpy()
        records.append({'model': model, 'batch': int(batch), 'samples': len(times),
                        'median_ms_batch': np.median(times), 'mean_ms_batch': times.mean(),
                        'std_ms_batch': times.std(ddof=1), 'p95_ms_batch': np.percentile(times, 95),
                        'median_ms_image': np.median(times) / batch,
                        'throughput_images_s': 1000 * batch / times.mean()})
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--outputs', type=Path, required=True, help='The outputs/ folder for this checkpoint.')
    parser.add_argument('--cache', type=Path, default=ROOT / 'p3_cache')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--warmup', type=int, default=30)
    parser.add_argument('--repeats', type=int, default=100)
    parser.add_argument('--sessions', type=int, default=3)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != 'cuda' or args.warmup < 0 or args.repeats < 2 or args.sessions < 1:
        parser.error('Use a CUDA device, warmup >= 0, repeats >= 2, and sessions >= 1.')
    selection = json.loads((args.outputs / 'selection.json').read_text())
    digest = sha256(args.checkpoint)
    if selection['checkpoint_sha256'] != digest:
        raise ValueError('The checkpoint does not match the selected checkpoint in --outputs.')
    configure_runtime(device, selection['seed'])
    data, _, _ = load_p3(args.cache)
    model = load_checkpoint(args.checkpoint).to(device)
    inputs = torch.stack([data['test'][index][0] for index in range(32)]).to(device)
    properties = torch.cuda.get_device_properties(device)
    uuid = 'GPU-' + str(properties.uuid).removeprefix('GPU-')
    free, total = torch.cuda.mem_get_info(device)
    now = datetime.now(timezone.utc)

    trials, activity = [], []
    for session in range(1, args.sessions + 1):
        for batch in ((1, 32) if session % 2 else (32, 1)):
            before = other_gpu_processes(uuid)
            times = measure(model, inputs[:batch], device, args.warmup, args.repeats)
            after = other_gpu_processes(uuid)
            activity.append({'session': session, 'batch': batch, 'before': before, 'after': after})
            trials.extend({'model': 'E7 T', 'batch': batch, 'session': session, 'trial': index, 'ms_batch': ms}
                          for index, ms in enumerate(times, 1))

    samples = [sample for record in activity for sample in (record['before'], record['after']) if sample is not None]
    metadata = {'status': 'COMPLETE', 'gpu': properties.name, 'precision': 'FP32',
                'started_utc': now.isoformat(), 'checkpoint_sha256': digest,
                'scope': 'E7-T model forward on resident P3 test inputs', 'batch_sizes': [1, 32],
                'warmup': args.warmup, 'repeats': args.repeats, 'sessions': args.sessions,
                'tf32': False, 'amp': False, 'free_gib_at_start': free / 2**30, 'total_gib': total / 2**30,
                'activity_sample_scope': 'before/after each timing session', 'allow_shared_gpu': True,
                'samples_with_other_gpu_activity': sum(bool(sample) for sample in samples),
                'gpu_activity_samples': len(samples), 'activity': activity}
    directory = args.outputs / 'latency' / now.strftime('%Y%m%dT%H%M%S_%fZ')
    directory.mkdir(parents=True, exist_ok=False)
    frame = pd.DataFrame(trials)
    frame.to_csv(directory / 'latency_trials.csv', index=False)
    summarize(frame).to_csv(directory / 'latency_summary.csv', index=False)
    write_json(directory / 'environment.json', metadata)
    print(f'Saved E7-T CUDA latency to {directory.resolve()}')


if __name__ == '__main__':
    main()

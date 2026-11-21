"""Train only PneuNet E7-T on the bundled P3 cache. Run with --smoke to check one update."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import shutil

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix, roc_auc_score
from torch import nn
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.utils.data import DataLoader, Dataset

from pneunet_e7_t import build_model, load_checkpoint
from report_e7_t import report_checkpoint_path, write_json, write_report


ROOT = Path(__file__).resolve().parent
SPLIT_SIZES = {'train': 4646, 'val': 586, 'test': 624}
CLASSES = ['NORMAL', 'PNEUMONIA']


def sha256(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


class P3Dataset(Dataset):
    def __init__(self, images, rows):
        self.images = images
        self.rows = rows.reset_index(drop=True)
        self.labels = rows.y.to_numpy(dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = torch.from_numpy(self.images[index].astype(np.float32)[None])
        return image, torch.tensor([self.labels[index]], dtype=torch.float32)


def load_p3(cache):
    manifest_path = cache / 'manifest.json'
    if not manifest_path.is_file():
        raise FileNotFoundError(f'{manifest_path} is missing. Copy the original p3_cache/ into the repo or pass --cache.')
    cache_sha256 = sha256(manifest_path)
    if cache_sha256 != sha256(ROOT / 'data' / 'p3_cache_manifest.json'):
        raise ValueError('The cache manifest differs from the archived P3 train/validation/test cache.')
    manifest = json.loads(manifest_path.read_text())
    index = cache / 'source_index.csv'
    if sha256(index) != manifest['index_sha256']:
        raise ValueError('The P3 source index checksum differs from its manifest.')
    rows = pd.read_csv(index)
    if not rows.source_index.is_unique or set(rows['split']) != set(SPLIT_SIZES):
        raise ValueError('Unexpected P3 source indices or splits.')
    datasets = {}
    for split, count in SPLIT_SIZES.items():
        path = cache / f'{split}.npy'
        if sha256(path) != manifest['arrays'][split]['sha256']:
            raise ValueError(f'The P3 {split} cache checksum differs from its manifest.')
        images = np.load(path, mmap_mode='r', allow_pickle=False)
        split_rows = rows[rows['split'] == split]
        if (images.shape != (count, 224, 224) or images.dtype != np.uint8
                or len(split_rows) != count or not split_rows.y.isin([0, 1]).all()):
            raise ValueError(f'Invalid images or labels for P3 {split}.')
        datasets[split] = P3Dataset(images, split_rows)

    config = dict(manifest['original_input_provenance']['training_config'])
    config['response_normalization'] = 'pre_pool_GRN'
    counts = np.bincount(datasets['train'].labels, minlength=2)
    weights = len(datasets['train']) / (2.0 * counts)
    np.testing.assert_array_equal(counts, config['class_counts'])
    np.testing.assert_array_equal(weights, config['class_weights'])
    if (config['seed'], config['batch_size'], config['image_size'], config['threshold']) != (42, 32, 224, 0.5):
        raise ValueError('Unexpected training configuration for this E7-T experiment.')
    return datasets, config, cache_sha256


def set_seed(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)


def configure_runtime(device, seed):
    if device.type == 'cuda':
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable. Use --device cpu for a CPU run.')
        torch.cuda.set_device(device)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    set_seed(seed, device)


def make_loader(dataset, config, device, *, shuffle=False):
    return DataLoader(dataset, batch_size=config['batch_size'], shuffle=shuffle,
                      num_workers=0, pin_memory=device.type == 'cuda', drop_last=False,
                      generator=torch.Generator().manual_seed(config['seed']))


def kernel_l2(model, coefficient):
    terms = [layer.weight.square().sum() for layer in model.modules()
             if isinstance(layer, (nn.Conv2d, nn.Linear))]
    return coefficient * torch.stack(terms).sum()


def weighted_bce(logits, targets, weights):
    logits, targets = logits.reshape(-1), targets.reshape(-1)
    losses = binary_cross_entropy_with_logits(logits, targets, reduction='none')
    return (losses * weights[targets.long()]).mean()


def update_batch(model, batch, optimizer, config, weights, device):
    images, labels = (tensor.to(device, non_blocking=True) for tensor in batch)
    optimizer.zero_grad(set_to_none=True)
    logits = model(images)
    loss = weighted_bce(logits, labels, weights) + kernel_l2(model, config['l2'])
    if not torch.isfinite(loss):
        raise RuntimeError('Training loss is not finite.')
    loss.backward()
    optimizer.step()
    return float(loss.detach()), len(images)


def train_one_epoch(model, loader, optimizer, config, weights, device):
    model.train()
    loss_total = 0.0
    for batch in loader:
        loss, count = update_batch(model, batch, optimizer, config, weights, device)
        loss_total += loss * count
    return loss_total / len(loader.dataset)


@torch.inference_mode()
def evaluate(model, loader, config, weights, device):
    model.eval()
    penalty = float(kernel_l2(model, config['l2']))
    loss_total = 0.0
    probabilities = []
    for images, labels in loader:
        logits = model(images.to(device, non_blocking=True))
        loss = float(weighted_bce(logits, labels.to(device, non_blocking=True), weights)) + penalty
        loss_total += loss * len(images)
        probabilities.append(logits.sigmoid().flatten().cpu().numpy())
    scores = np.concatenate(probabilities)
    labels = loader.dataset.labels
    predictions = (scores >= config['threshold']).astype(np.int64)
    report = classification_report(labels, predictions, labels=[0, 1], target_names=CLASSES,
                                   output_dict=True, zero_division=0)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = map(int, matrix.ravel())
    has_both_classes = np.unique(labels).size == 2
    metrics = {
        'loss': loss_total / len(labels), 'macro_f1': report['macro avg']['f1-score'],
        'accuracy': report['accuracy'], 'samples': len(labels), 'threshold': config['threshold'],
        'roc_auc': float(roc_auc_score(labels, scores)) if has_both_classes else None,
        'pr_auc': float(average_precision_score(labels, scores)) if has_both_classes else None,
        'tn_normal': tn, 'fp_normal': fp, 'fn_pneumonia': fn, 'tp_pneumonia': tp,
    }
    for name in CLASSES:
        for metric in ('precision', 'recall', 'f1-score'):
            metrics[f"{metric.replace('-score', '')}_{name.lower()}"] = report[name][metric]
    return {**metrics, 'report': report, 'confusion_matrix': matrix.tolist()}, scores, predictions


def save_checkpoint(path, payload):
    temporary = path.with_suffix('.pt.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def save_history(output, history):
    temporary = output / 'history.csv.tmp'
    pd.DataFrame(history).to_csv(temporary, index=False)
    temporary.replace(output / 'history.csv')


def capture_rng(loader, device):
    name, keys, *tail = np.random.get_state()
    return {'python': random.getstate(), 'numpy': (name, keys.tolist(), *tail),
            'torch': torch.get_rng_state(), 'loader': loader.generator.get_state(),
            'cuda': torch.cuda.get_rng_state(device) if device.type == 'cuda' else None}


def restore_rng(state, loader, device):
    random.setstate(state['python'])
    name, keys, *tail = state['numpy']
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), *tail))
    torch.set_rng_state(state['torch'])
    loader.generator.set_state(state['loader'])
    if device.type == 'cuda':
        torch.cuda.set_rng_state(state['cuda'], device)


def export_run(model, loaders, config, weights, device, run, best_epoch, history, cache_sha256, role):
    output = run / 'outputs'
    evaluations = {}
    for split, loader in loaders.items():
        if split == 'train':
            loader = make_loader(loader.dataset, config, device)
        measured, scores, decisions = evaluate(model, loader, config, weights, device)
        report = measured.pop('report')
        matrix = measured.pop('confusion_matrix')
        predictions = loader.dataset.rows[['source_index', 'y']].rename(columns={'y': 'y_true'}).assign(
            probability=scores, prediction=decisions,
        )
        evaluations[split] = (measured, report, matrix, predictions)
        print(f'{split}: Macro-F1 {measured["macro_f1"]:.6f} on {len(scores)} images', flush=True)

    metrics = {split: record[0] for split, record in evaluations.items()}
    checkpoint = run / 'best.pt'
    selection = {
        'status': 'COMPLETE', 'candidate': 'T', 'seed': config['seed'],
        'parameters': sum(parameter.numel() for parameter in model.parameters()),
        'best_epoch': best_epoch, 'epochs_completed': len(history) if history else None,
        'validation_macro_f1': metrics['val']['macro_f1'], 'test_macro_f1': metrics['test']['macro_f1'],
        'checkpoint': report_checkpoint_path(checkpoint, output), 'checkpoint_sha256': sha256(checkpoint),
        'checkpoint_metric': 'validation_macro_f1', 'evaluation_role': role,
        'test_used_for_selection': role == 're-evaluation of the archived T checkpoint',
        'cache_manifest_sha256': cache_sha256,
    }
    write_report(output, evaluations, selection, config, history, model, loaders['test'].dataset, device)
    write_json(output / 'result.json', {**selection, 'metrics': metrics})


def run_training(model, optimizer, loaders, config, weights, device, output, epochs, signature, resume):
    train = loaders['train']
    validation = loaders['val']
    state = {'history': [], 'best': None, 'lowest_val_loss': float('inf'), 'stale': 0}
    if resume:
        state = torch.load(output / 'last.pt', map_location='cpu', weights_only=True)
        if state['signature'] != signature:
            raise ValueError('Checkpoint, cache, source code or training environment differs from this run.')
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        restore_rng(state['rng'], train, device)
    if epochs < len(state['history']):
        raise ValueError('--epochs is a total epoch limit and must include completed epochs.')
    output_reports = output / 'outputs'
    output_reports.mkdir(exist_ok=True)

    for epoch in range(len(state['history']) + 1, epochs + 1):
        if state['stale'] >= config['early_stopping_patience']:
            break
        loss = train_one_epoch(model, train, optimizer, config, weights, device)
        val, _, _ = evaluate(model, validation, config, weights, device)
        if not np.isfinite([loss, val['loss'], val['macro_f1']]).all():
            raise RuntimeError(f'Non-finite metrics at epoch {epoch}.')
        state['history'].append({'epoch': epoch, 'train_loss': loss, 'val_loss': val['loss'],
                                 'val_macro_f1': val['macro_f1'], 'val_roc_auc': val['roc_auc']})
        improved = state['best'] is None or val['macro_f1'] > state['best']['val_macro_f1']
        if improved:
            state['best'] = {'model_name': 'PneuNetE7T', 'config': config, 'epoch': epoch,
                             'val_macro_f1': val['macro_f1'],
                             'state_dict': {k: v.detach().to('cpu', copy=True) for k, v in model.state_dict().items()}}
        if val['loss'] < state['lowest_val_loss']:
            state['lowest_val_loss'], state['stale'] = val['loss'], 0
        else:
            state['stale'] += 1
        save_checkpoint(output / 'last.pt', {
            'signature': signature, **state, 'model': model.state_dict(),
            'optimizer': optimizer.state_dict(), 'rng': capture_rng(train, device),
        })
        if improved:
            save_checkpoint(output / 'best.pt', state['best'])
        save_history(output_reports, state['history'])
        print(f'Epoch {epoch:02d}  train loss {loss:.5f}  val loss {val["loss"]:.5f}  '
              f'val Macro-F1 {val["macro_f1"]:.5f}', flush=True)

    best = state['best']
    save_checkpoint(output / 'best.pt', best)
    model.load_state_dict(best['state_dict'], strict=True)
    export_run(model, loaders, config, weights, device, output, best['epoch'], state['history'],
               signature['cache_manifest_sha256'], 'fixed architecture reproduction')
    print(f'Best epoch {best["epoch"]} by validation Macro-F1.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=ROOT / 'p3_cache')
    parser.add_argument('--epochs', type=int, default=50, help='Total epoch limit, including any resumed epochs.')
    parser.add_argument('--device', default='cuda:0', help='Training device, for example cuda:0 or cpu.')
    destinations = parser.add_mutually_exclusive_group()
    destinations.add_argument('--output', type=Path, help='New run directory. Existing directories are rejected.')
    destinations.add_argument('--resume', type=Path, help='Continue a run made by this script from its last.pt.')
    parser.add_argument('--smoke', action='store_true', help='Run one real train batch with an optimizer step, then exit.')
    parser.add_argument('--evaluate', type=Path, metavar='BEST_PT', help='Export a report from an existing checkpoint.')
    parser.add_argument('--history', type=Path, help='Original history.csv for an evaluation-only run.')
    args = parser.parse_args()
    if args.smoke and (args.output or args.resume or args.evaluate):
        parser.error('--smoke runs in memory and does not accept --output, --resume or --evaluate.')
    if (args.evaluate and args.resume) or (args.history and not args.evaluate):
        parser.error('--evaluate is separate from --resume, and --history requires --evaluate.')

    datasets, config, cache_sha256 = load_p3(args.cache)
    if not 1 <= args.epochs <= config['epochs']:
        parser.error(f'--epochs must be between 1 and {config["epochs"]}.')
    device = torch.device(args.device)
    if device.type not in ('cpu', 'cuda'):
        parser.error('--device must be cpu or cuda[:index].')
    configure_runtime(device, config['seed'])
    loaders = {split: make_loader(dataset, config, device, shuffle=split == 'train')
               for split, dataset in datasets.items()}
    weights = torch.tensor(config['class_weights'], dtype=torch.float32, device=device)
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output = args.resume or args.output or ROOT / 'runs' / f'E7_T_{timestamp}'
    if args.evaluate:
        checkpoint = torch.load(args.evaluate, map_location='cpu', weights_only=True)
        actual_config = checkpoint.get('signature', checkpoint).get('config')
        if actual_config != config:
            raise ValueError('The checkpoint training configuration differs from the P3 protocol.')
        model = load_checkpoint(args.evaluate).to(device)
        history = pd.read_csv(args.history).to_dict('records') if args.history else []
        output.mkdir(parents=True, exist_ok=False)
        shutil.copy2(args.evaluate, output / 'best.pt')
        role = 're-evaluation of the archived T checkpoint' if 'signature' in checkpoint else 'checkpoint re-evaluation'
        export_run(model, loaders, config, weights, device, output, checkpoint['epoch'], history, cache_sha256, role)
        print(f'Report directory: {(output / "outputs").resolve()}')
        return

    model = build_model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'],
                                 betas=(config['beta_1'], config['beta_2']), eps=config['epsilon'])
    set_seed(config['seed'], device)
    print(f'E7-T on {device}: {sum(p.numel() for p in model.parameters()):,} parameters', flush=True)

    if args.smoke:
        loss, count = update_batch(model, next(iter(loaders['train'])), optimizer, config, weights, device)
        if count != config['batch_size'] or not all(
            p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.parameters()
        ):
            raise RuntimeError('Smoke check found a bad batch size or non-finite gradients.')
        print(f'Smoke passed: {count} P3 train images, one FP32 optimizer step, loss {loss:.6f}.')
        return

    if not args.resume:
        output.mkdir(parents=True, exist_ok=False)
    signature = {'config': config, 'cache_manifest_sha256': cache_sha256, 'device': str(device),
                 'torch_version': str(torch.__version__),
                 'model_source_sha256': sha256(ROOT / 'pneunet_e7_t.py'),
                 'trainer_source_sha256': sha256(Path(__file__)),
                 'report_source_sha256': sha256(ROOT / 'report_e7_t.py')}
    run_training(model, optimizer, loaders, config, weights, device, output, args.epochs, signature, bool(args.resume))
    print(f'Run directory: {output.resolve()}')


if __name__ == '__main__':
    main()

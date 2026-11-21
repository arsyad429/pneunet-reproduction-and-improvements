"""Write the CSV, JSON and PNG artifacts displayed by E7B15_Final.ipynb."""

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parent
CLASSES = ('NORMAL', 'PNEUMONIA')


def write_json(path, content):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(content, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def plot_confusion(matrix, title, path):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(matrix, cmap='Blues')
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=CLASSES, yticklabels=CLASSES,
           xlabel='Predicted class', ylabel='Actual class', title=title)
    for row in range(2):
        for column in range(2):
            color = 'white' if matrix[row][column] > max(map(max, matrix)) * 0.55 else 'black'
            ax.text(column, row, matrix[row][column], ha='center', va='center', color=color)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_learning_curve(history, path):
    if not history:
        return
    frame = pd.DataFrame(history)
    fig, (loss, score) = plt.subplots(1, 2, figsize=(9, 3.5))
    loss.plot(frame.epoch, frame.train_loss, label='Train')
    loss.plot(frame.epoch, frame.val_loss, label='Validation')
    loss.set(xlabel='Epoch', ylabel='Weighted BCE + L2')
    loss.legend()
    score.plot(frame.epoch, frame.val_macro_f1)
    score.set(xlabel='Epoch', ylabel='Validation Macro-F1', ylim=(0, 1))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _gradcam(model, image, target):
    captured = []
    hook = model.context_refine.response_norm.register_forward_hook(
        lambda _layer, _inputs, output: captured.append(output)
    )
    try:
        with torch.enable_grad():
            logit = model(image).reshape(())
            feature = captured[0]
            gradient, = torch.autograd.grad(logit if target == 1 else -logit, feature)
            weight = gradient.mean(dim=(2, 3), keepdim=True)
            heatmap = (weight * feature).sum(dim=1, keepdim=True).relu()
            heatmap = F.interpolate(heatmap, size=image.shape[-2:], mode='bilinear', align_corners=False)
            heatmap = heatmap / heatmap.max().clamp_min(1e-6)
        return heatmap.detach().squeeze().cpu().numpy()
    finally:
        hook.remove()


def write_gradcam(model, dataset, predictions, device, output):
    cases = []
    for actual, predicted, group in ((0, 1, 'FP'), (1, 0, 'FN'), (0, 0, 'TN'), (1, 1, 'TP')):
        rows = predictions[(predictions.y_true == actual) & (predictions.prediction == predicted)]
        if not rows.empty:
            index = int(rows.probability.idxmax() if predicted == 1 else rows.probability.idxmin())
            cases.append((index, group))

    rows = []
    model.eval()
    for target, label in enumerate(CLASSES):
        fig, axes = plt.subplots(len(cases), 2, figsize=(6, 2.7 * len(cases)), squeeze=False)
        for (index, group), (original, overlay) in zip(cases, axes):
            image, _ = dataset[index]
            source = predictions.iloc[index]
            pixels = image[0].numpy()
            heatmap = _gradcam(model, image[None].to(device), target)
            original.imshow(pixels, cmap='gray', vmin=0, vmax=255)
            original.set_title(f'P3 image {int(source.source_index)} · {group}')
            overlay.imshow(pixels, cmap='gray', vmin=0, vmax=255)
            overlay.imshow(heatmap, cmap='jet', vmin=0, vmax=1, alpha=heatmap * 0.65)
            overlay.set_title(f'{label} · P(PNEUMONIA) {source.probability:.3f}')
            original.axis('off')
            overlay.axis('off')
            rows.append({'source_index': int(source.source_index), 'case': group, 'target': label,
                         'target_layer': 'context_refine.response_norm',
                         'probability_pneumonia': float(source.probability)})
        fig.tight_layout()
        fig.savefig(output / f'gradcam_p3_target_{target}.png', dpi=140)
        plt.close(fig)
    pd.DataFrame(rows).to_csv(output / 'gradcam_p3_cases.csv', index=False)


def write_report(output, evaluations, selection, config, history, model, test_dataset, device):
    """Export evaluated splits and reference comparisons for one specific checkpoint."""
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'config.json', config)
    if history:
        pd.DataFrame(history).to_csv(output / 'history.csv', index=False)
        plot_learning_curve(history, output / 'learning_curve.png')

    for split, (metrics, report, matrix, predictions) in evaluations.items():
        write_json(output / f'{split}_metrics.json', metrics)
        write_json(output / f'{split}_report.json', report)
        pd.DataFrame(matrix, index=CLASSES, columns=CLASSES).to_csv(output / f'{split}_confusion_matrix.csv')
        predictions.to_csv(output / f'{split}_predictions.csv', index=False)
        plot_confusion(matrix, split.capitalize(), output / f'{split}_confusion_matrix.png')

    test, _, _, predictions = evaluations['test']
    # FLOPs describe the fixed T architecture, so they do not change with seed42 weights.
    comparison = [{'model': 'E7 T', 'parameters': selection['parameters'], **test,
                   'counted_flops': 8_866_300_928, 'result_source': 'current checkpoint'}]
    baselines = pd.read_csv(ROOT / 'data' / 'baseline_classification_metrics.csv').to_dict('records')
    comparison.extend({**row, 'result_source': 'archived P3 baseline'} for row in baselines)
    pd.DataFrame(comparison).to_csv(output / 'final_comparison.csv', index=False, na_rep='N/A')
    write_gradcam(model, test_dataset, predictions, device, output)
    write_json(output / 'selection.json', selection)


def report_checkpoint_path(checkpoint, output):
    return os.path.relpath(checkpoint.resolve(), output.resolve())

"""Rebuild the frozen E7-T P3 cache from the original chest X-rays."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from threading import local
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT / 'data' / 'p3_cache_manifest.json'
SPLITS = ('train', 'val', 'test')
CLASSES = ('NORMAL', 'PNEUMONIA')
IMAGE_EXTENSIONS = {'.jpeg', '.jpg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
SIZE = 224


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def source_index(dataset, expected_counts):
    rows = []
    for split in SPLITS:
        for label, name in enumerate(CLASSES):
            directory = dataset / split / name
            if not directory.is_dir():
                raise FileNotFoundError(f'Missing dataset class directory: {directory}')
            for path in sorted(directory.rglob('*')):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    relative = path.relative_to(dataset).as_posix()
                    rows.append({'relpath': relative, 'split': split, 'label': name, 'y': label, 'path': relative})
    counts = Counter(row['split'] for row in rows)
    if counts != expected_counts:
        raise ValueError(f'Expected dataset splits {expected_counts}, found {dict(counts)}.')
    return pd.DataFrame(rows).rename_axis('source_index')


def augmentation_plan(index, seed):
    """Assign one fixed augmentation per training image, stratified by class."""
    plan = index[['path', 'split', 'label', 'y']].copy()
    plan['aug_mode'] = 0
    plan['rotation_degrees'] = 0.0
    plan['gaussian_sigma'] = 0.0
    train = plan.index[plan.split == 'train'].to_numpy()
    labels = plan.loc[train, 'y'].to_numpy()
    modes = np.empty(len(train), dtype=np.uint8)
    rng = np.random.default_rng(seed)
    for label in np.unique(labels):
        positions = rng.permutation(np.flatnonzero(labels == label))
        ideal = len(positions) * np.array([0.50, 0.25, 0.25])
        quotas = np.floor(ideal).astype(int)
        remaining = len(positions) - quotas.sum()
        order = np.argsort(-(ideal - quotas), kind='stable')
        quotas[order[:remaining]] += 1
        modes[positions] = np.repeat(np.arange(3, dtype=np.uint8), quotas)
    plan.loc[train, 'aug_mode'] = modes.astype(int)
    for index, mode in zip(train, modes):
        if mode == 1:
            plan.at[index, 'rotation_degrees'] = float(rng.uniform(-15.0, 15.0))
        elif mode == 2:
            plan.at[index, 'gaussian_sigma'] = float(rng.uniform(0.5, 1.5))
    return plan


def read_gray(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None or image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError(f'Cannot read grayscale image: {path}')
    return image


class CachedMasks:
    def __init__(self, directory, provenance):
        self.directory = directory
        observed = json.loads((directory / '_provenance.json').read_text())
        if observed != provenance:
            raise ValueError(f'Mask provenance differs from the frozen P3 protocol: {directory}')

    def __call__(self, row, image):
        name = hashlib.sha256(row.relpath.encode()).hexdigest()[:24] + '.png'
        path = self.directory / row.split / row.label / name
        mask = read_gray(path)
        if mask.shape != image.shape or ((mask != 0) & (mask != 255)).any():
            raise ValueError(f'Mask must be binary 0/255 with the original image shape: {path}')
        return mask > 0


class UNetResNet34Masks:
    def __init__(self, weights, device, provenance):
        import torch

        if sha256(weights) != provenance['release_weights_sha256']:
            raise ValueError('Use the official resnet34.pth release checkpoint recorded in the P3 manifest.')
        try:
            from lungs_segmentation import unet
        except ModuleNotFoundError as error:
            raise RuntimeError('Install lungs-segmentation==1.1.2 with --no-deps for this option.') from error

        self.device = torch.device(device)
        if self.device.type not in ('cpu', 'cuda'):
            raise ValueError('--device must be cpu or cuda[:index].')
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable. Use --device cpu or the archived mask directory.')
        torch.set_num_threads(2)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        # The archived masks were generated before the classifier disabled TF32.
        # Matching cuDNN's mask-inference setting preserves the 0.20 threshold boundary.
        torch.backends.cudnn.allow_tf32 = self.device.type == 'cuda'
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)

        # The full release checkpoint contains encoder and decoder weights.
        # Instantiate the official topology without downloading encoder weights.
        encoder = dict(unet.encoder_params['resnet34'])
        encoder['init_op'] = lambda pretrained: unet.resnet.resnet34(in_channels=3, pretrained=False)
        encoder['url'] = None
        with patch.dict(unet.encoder_params, {'resnet34': encoder}):
            model = unet.Resnet(seg_classes=2, backbone_arch='resnet34')
        state = torch.load(weights, map_location='cpu', weights_only=True)
        model.load_state_dict(state, strict=True)
        digest = hashlib.sha256()
        for name, tensor in model.state_dict().items():
            digest.update(name.encode())
            digest.update(tensor.numpy().tobytes())
        if digest.hexdigest() != provenance['checkpoint_sha256']:
            raise ValueError('The effective U-Net parameters differ from the archived model.')
        self.torch = torch
        self.model = model.to(self.device).eval()
        self.threshold = provenance['threshold']

    def __call__(self, row, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB).astype(np.float32)
        lo, hi = rgb.min(), rgb.max()
        rgb = (rgb - lo) / (hi - lo) if hi > lo else np.zeros_like(rgb)
        resized = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_LINEAR)
        batch = self.torch.from_numpy(resized.transpose(2, 0, 1).copy())[None].to(self.device)
        with self.torch.inference_mode():
            probabilities = self.model(batch).sigmoid()[0].cpu().numpy()
        if probabilities.shape != (2, 512, 512):
            raise ValueError(f'Unexpected U-Net output for {row.relpath}: {probabilities.shape}')
        union = np.any(probabilities > self.threshold, axis=0).astype(np.uint8)
        height, width = image.shape
        return cv2.resize(union, (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)


def letterbox(image):
    height, width = image.shape
    scale = SIZE / max(height, width)
    new_width = max(1, min(SIZE, int(round(width * scale))))
    new_height = max(1, min(SIZE, int(round(height * scale))))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_width, new_height), interpolation=interpolation)
    result = np.zeros((SIZE, SIZE), dtype=np.uint8)
    top, left = (SIZE - new_height) // 2, (SIZE - new_width) // 2
    result[top:top + new_height, left:left + new_width] = resized
    return result


def preprocess(image, mask, augmentation, clahe):
    if not mask.any() or mask.all():
        raise ValueError(f'Empty or full lung mask: {augmentation.path}')
    result = letterbox(np.where(mask, clahe.apply(image), 0).astype(np.uint8))
    if augmentation.aug_mode == 1:
        transform = cv2.getRotationMatrix2D((SIZE / 2, SIZE / 2), float(augmentation.rotation_degrees), 1.0)
        result = cv2.warpAffine(result, transform, (SIZE, SIZE), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    elif augmentation.aug_mode == 2:
        sigma = float(augmentation.gaussian_sigma)
        result = cv2.GaussianBlur(result, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)
    return result


def write_split(split, index, plan, dataset, mask_provider, path, expected, workers):
    rows = list(index[index.split == split].itertuples())
    thread_state = local()

    def process_row(row):
        if not hasattr(thread_state, 'clahe'):
            thread_state.clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        image = read_gray(dataset / row.relpath)
        return preprocess(image, mask_provider(row, image), plan.iloc[row.Index], thread_state.clahe)

    images = np.lib.format.open_memmap(path, mode='w+', dtype=np.uint8, shape=tuple(expected['shape']))
    try:
        if workers == 1:
            for position, image in enumerate(map(process_row, rows), 1):
                images[position - 1] = image
                if position % 250 == 0 or position == len(rows):
                    print(f'{split}: {position}/{len(rows)} images', flush=True)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for first in range(0, len(rows), 128):
                    # Bound buffered results and retain the archive row order.
                    for position, image in enumerate(pool.map(process_row, rows[first:first + 128]), first + 1):
                        images[position - 1] = image
                        if position % 250 == 0 or position == len(rows):
                            print(f'{split}: {position}/{len(rows)} images', flush=True)
        images.flush()
    finally:
        del images
    digest = sha256(path)
    if digest != expected['sha256']:
        raise ValueError(f'{split}.npy checksum mismatch. Expected {expected["sha256"]}, found {digest}. '
                         'Check the raw images, masks, NumPy/OpenCV versions and GPU mask inference environment.')
    print(f'{split}.npy: SHA-256 matches the archived P3 cache.', flush=True)


def build_cache(dataset, output, mask_dir=None, weights=None, device='cuda:0', workers=None):
    output = output.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'{output} already exists. Choose a new --output directory.')
    manifest = json.loads(REFERENCE.read_text())
    provenance = manifest['original_input_provenance']
    index = source_index(dataset, provenance['counts'])
    plan = augmentation_plan(index, provenance['training_config']['seed'])
    workers = (8 if mask_dir is not None else 1) if workers is None else workers
    if workers < 1 or (weights is not None and workers != 1):
        raise ValueError('Use at least one worker; U-Net generation requires exactly one worker.')
    cv2.setNumThreads(1 if workers > 1 else 2)
    cv2.ocl.setUseOpenCL(False)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.p3_build_', dir=output.parent) as temporary:
        cache = Path(temporary) / 'cache'
        cache.mkdir()
        csv_file = cache / 'source_index.csv'
        index.to_csv(csv_file, lineterminator='\n')
        if sha256(csv_file) != manifest['index_sha256']:
            raise ValueError('Dataset filenames or splits differ from the archived P3 source index.')
        plan.to_csv(cache / 'static_augmentation_plan.csv', index=False, lineterminator='\n')
        provider = (CachedMasks(mask_dir, provenance['mask_provenance']) if mask_dir is not None
                    else UNetResNet34Masks(weights, device, provenance['mask_provenance']))
        for split in SPLITS:
            write_split(split, index, plan, dataset, provider, cache / f'{split}.npy', manifest['arrays'][split], workers)
        shutil.copyfile(REFERENCE, cache / 'manifest.json')
        if output.exists() or output.is_symlink():
            raise FileExistsError(output)
        cache.rename(output)
    print(f'P3 cache ready: {output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True, help='chest_xray directory containing train, val and test.')
    masks = parser.add_mutually_exclusive_group(required=True)
    masks.add_argument('--mask-dir', type=Path, help='Original unet_resnet34 mask cache with _provenance.json.')
    masks.add_argument('--unet-weights', type=Path, help='Official resnet34.pth for building masks from the X-rays.')
    parser.add_argument('--device', default='cuda:0', help='Mask generation device. Cached masks need only the CPU.')
    parser.add_argument('--workers', type=int, help='CPU workers for preprocessing cached masks (default: 8).')
    parser.add_argument('--output', type=Path, default=ROOT / 'p3_cache')
    args = parser.parse_args()
    build_cache(args.dataset, args.output, mask_dir=args.mask_dir, weights=args.unet_weights,
                device=args.device, workers=args.workers)


if __name__ == '__main__':
    main()

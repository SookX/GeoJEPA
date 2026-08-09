#!/usr/bin/env python
"""Apply small compatibility patches needed by the GeoJEPA LeWM harness."""

from __future__ import annotations

import argparse
from pathlib import Path


PIXEL_PREPROCESSOR = '''\n\nclass PixelPreprocessor:\n    """Picklable image preprocessor for LeWM pixel sequences."""\n\n    def __init__(self, source: str, target: str, img_size: int = 224):\n        self.source = source\n        self.target = target\n        self.img_size = img_size\n        stats = dt.dataset_stats.ImageNet\n        self.mean = torch.tensor(stats["mean"], dtype=torch.float32).view(1, 3, 1, 1)\n        self.std = torch.tensor(stats["std"], dtype=torch.float32).view(1, 3, 1, 1)\n\n    def __call__(self, sample):\n        pixels = torch.as_tensor(sample[self.source]).float()\n        if pixels.ndim != 4:\n            raise ValueError(f"Expected {self.source} to have shape (T,H,W,C) or (T,C,H,W), got {tuple(pixels.shape)}")\n\n        if pixels.shape[-1] in (1, 3):\n            pixels = pixels.permute(0, 3, 1, 2)\n        if pixels.shape[1] == 1:\n            pixels = pixels.expand(-1, 3, -1, -1)\n\n        if pixels.max() > 2:\n            pixels = pixels / 255.0\n        if pixels.shape[-2:] != (self.img_size, self.img_size):\n            pixels = F.interpolate(\n                pixels,\n                size=(self.img_size, self.img_size),\n                mode="bilinear",\n                align_corners=False,\n                antialias=True,\n            )\n        sample[self.target] = (pixels - self.mean.to(pixels.device)) / self.std.to(pixels.device)\n        return sample\n'''

OLD_PREPROCESSOR = '''def get_img_preprocessor(source: str, target: str, img_size: int = 224):\n    imagenet_stats = dt.dataset_stats.ImageNet\n    to_image = dt.transforms.ToImage(**imagenet_stats, source=source, target=target)\n    resize = dt.transforms.Resize(img_size, source=source, target=target)\n    return dt.transforms.Compose(to_image, resize)'''

NEW_PREPROCESSOR = '''def get_img_preprocessor(source: str, target: str, img_size: int = 224):\n    return PixelPreprocessor(source=source, target=target, img_size=img_size)'''


def patch_utils(lewm_dir: Path) -> bool:
    path = lewm_dir / "utils.py"
    text = path.read_text(encoding="utf-8")
    changed = False
    if "import torch.nn.functional as F" not in text:
        text = text.replace("import torch\n", "import torch\nimport torch.nn.functional as F\n")
        changed = True
    if "class PixelPreprocessor" not in text:
        marker = "def get_img_preprocessor"
        text = text.replace(marker, PIXEL_PREPROCESSOR + "\n" + marker, 1)
        changed = True
    if OLD_PREPROCESSOR in text:
        text = text.replace(OLD_PREPROCESSOR, NEW_PREPROCESSOR)
        changed = True
    if changed:
        path.write_text(text, encoding="utf-8")
    return changed


def patch_train(lewm_dir: Path) -> bool:
    path = lewm_dir / "train.py"
    text = path.read_text(encoding="utf-8")
    changed = False
    if "import signal" not in text:
        text = text.replace("import os\n", "import os\nimport signal\n", 1)
        changed = True
    shim = '''\nfor _sig_name in ("SIGUSR1", "SIGUSR2", "SIGCONT"):\n    if not hasattr(signal, _sig_name):\n        setattr(signal, _sig_name, signal.SIGTERM)\n\n'''
    if "SIGUSR1" not in text.split("def lejepa_forward", 1)[0]:
        text = text.replace("\ndef lejepa_forward", shim + "\ndef lejepa_forward", 1)
        changed = True
    if changed:
        path.write_text(text, encoding="utf-8")
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lewm-dir", required=True)
    args = parser.parse_args()
    lewm_dir = Path(args.lewm_dir)
    if not (lewm_dir / "train.py").is_file() or not (lewm_dir / "utils.py").is_file():
        raise SystemExit(f"Not a LeWM source directory: {lewm_dir}")
    changed = []
    if patch_utils(lewm_dir):
        changed.append("utils.py")
    if patch_train(lewm_dir):
        changed.append("train.py")
    print("Patched " + ", ".join(changed) if changed else "LeWM compatibility patches already present")


if __name__ == "__main__":
    main()

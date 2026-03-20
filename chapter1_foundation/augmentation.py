"""3D Medical Image Augmentation for Brain MRI.

Comprehensive augmentation pipeline including:
- Random LR flip with hemisphere label swap
- Random affine (rotation ±10°, translation ±5px, scale ±10%)
- Elastic deformation (controlled amplitude)
- Gaussian noise (std up to 0.04)
- Bias field simulation
- Intensity/contrast adjustment
- Gamma correction
- Percentile clamping + normalization
"""
from __future__ import annotations
from typing import Dict
import numpy as np


class Compose3D:
    def __init__(self, transforms: list):
        self.transforms = transforms
    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        for t in self.transforms:
            sample = t(sample)
        return sample


class RandomFlipLR:
    """Left-right flip with hemisphere label swap.

    Uses REMAPPED contiguous indices (0-21), not raw FreeSurfer labels.
    """
    def __init__(self, p=0.5):
        self.p = p
        self.swap = {
            1:12, 12:1, 2:13, 13:2, 3:14, 14:3, 4:15, 15:4,
            5:16, 16:5, 6:17, 17:6, 7:18, 18:7, 9:19, 19:9,
            10:20, 20:10, 11:21, 21:11,
        }
    def __call__(self, s):
        if np.random.rand() > self.p:
            return s
        s["image"] = np.ascontiguousarray(s["image"][..., ::-1])
        if s.get("segmentation") is not None:
            seg = np.ascontiguousarray(s["segmentation"][..., ::-1])
            new = seg.copy()
            for a, b in self.swap.items():
                new[seg == a] = b
            s["segmentation"] = new
        return s


class RandomAffine3D:
    """Small affine: rotation ±deg, scale range, translation ±px."""
    def __init__(self, rot=10.0, scale=(0.90, 1.10), translate=5, p=0.5):
        self.rot, self.scale, self.translate, self.p = rot, scale, translate, p
    def __call__(self, s):
        if np.random.rand() > self.p:
            return s
        from scipy.ndimage import affine_transform
        shape = s["image"].shape
        center = np.array(shape) / 2.0
        angles = np.deg2rad(np.random.uniform(-self.rot, self.rot, 3))
        sc = np.random.uniform(*self.scale)
        c, sn = np.cos, np.sin
        Rx = np.array([[1,0,0],[0,c(angles[0]),-sn(angles[0])],[0,sn(angles[0]),c(angles[0])]])
        Ry = np.array([[c(angles[1]),0,sn(angles[1])],[0,1,0],[-sn(angles[1]),0,c(angles[1])]])
        Rz = np.array([[c(angles[2]),-sn(angles[2]),0],[sn(angles[2]),c(angles[2]),0],[0,0,1]])
        R = Rz @ Ry @ Rx / sc
        trans = np.random.uniform(-self.translate, self.translate, 3)
        off = center - R @ center + trans
        s["image"] = affine_transform(s["image"], R, offset=off, order=1, mode="constant", cval=0)
        if s.get("segmentation") is not None:
            s["segmentation"] = affine_transform(
                s["segmentation"].astype(np.float64), R, offset=off, order=0, mode="constant", cval=0
            ).astype(np.int64)
        return s


class RandomElasticDeform:
    """Smooth elastic deformation to simulate anatomical variability."""
    def __init__(self, alpha=8.0, sigma=4.0, p=0.3):
        self.alpha, self.sigma, self.p = alpha, sigma, p
    def __call__(self, s):
        if np.random.rand() > self.p:
            return s
        from scipy.ndimage import gaussian_filter, map_coordinates
        shape = s["image"].shape
        alpha = np.random.uniform(self.alpha * 0.5, self.alpha)
        dx = gaussian_filter(np.random.randn(*shape) * alpha, self.sigma, mode='constant')
        dy = gaussian_filter(np.random.randn(*shape) * alpha, self.sigma, mode='constant')
        dz = gaussian_filter(np.random.randn(*shape) * alpha, self.sigma, mode='constant')
        x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]),
                               np.arange(shape[2]), indexing='ij')
        coords = [np.clip(x+dx, 0, shape[0]-1),
                  np.clip(y+dy, 0, shape[1]-1),
                  np.clip(z+dz, 0, shape[2]-1)]
        s["image"] = map_coordinates(s["image"], coords, order=1, mode='constant', cval=0).astype(np.float32)
        if s.get("segmentation") is not None:
            s["segmentation"] = map_coordinates(
                s["segmentation"].astype(float), coords, order=0, mode='constant', cval=0
            ).astype(np.int64)
        return s


class RandomBiasField:
    def __init__(self, coeff=0.3, order=3, p=0.3):
        self.coeff, self.order, self.p = coeff, order, p
    def __call__(self, s):
        if np.random.rand() > self.p:
            return s
        shape = s["image"].shape
        grid = np.meshgrid(*[np.linspace(-1,1,d) for d in shape], indexing="ij")
        bias = np.zeros(shape, dtype=np.float64)
        for i in range(self.order+1):
            for j in range(self.order+1-i):
                k = self.order - i - j
                if k < 0: continue
                bias += np.random.uniform(-self.coeff, self.coeff) * grid[0]**i * grid[1]**j * grid[2]**k
        s["image"] = (s["image"] * np.exp(bias)).astype(np.float32)
        return s


class RandomNoise:
    def __init__(self, std=(0., 0.04), p=0.4):
        self.std, self.p = std, p
    def __call__(self, s):
        if np.random.rand() > self.p:
            return s
        noise_std = np.random.uniform(*self.std)
        s["image"] = (s["image"] + np.random.randn(*s["image"].shape).astype(np.float32) * noise_std)
        return s


class RandomIntensityShift:
    def __init__(self, shift=0.1, scale=(0.90, 1.10), p=0.4):
        self.shift, self.scale, self.p = shift, scale, p
    def __call__(self, s):
        if np.random.rand() > self.p:
            return s
        s["image"] = s["image"] * np.random.uniform(*self.scale) + np.random.uniform(-self.shift, self.shift)
        return s


class RandomGamma:
    def __init__(self, gamma=(0.80, 1.20), p=0.3):
        self.gamma, self.p = gamma, p
    def __call__(self, s):
        if np.random.rand() > self.p:
            return s
        g = np.random.uniform(*self.gamma)
        img = s["image"]
        mn, mx = img.min(), img.max()
        if mx - mn < 1e-8: return s
        s["image"] = np.power((img - mn) / (mx - mn), g) * (mx - mn) + mn
        return s


class ClampNormalize:
    def __init__(self, lo=1., hi=99.):
        self.lo, self.hi = lo, hi
    def __call__(self, s):
        img = s["image"]
        p_lo, p_hi = np.percentile(img, [self.lo, self.hi])
        img = np.clip(img, p_lo, p_hi)
        s["image"] = ((img - p_lo) / max(p_hi - p_lo, 1e-8)).astype(np.float32)
        return s


def get_train_augmentation() -> Compose3D:
    """Full training augmentation pipeline."""
    return Compose3D([
        RandomFlipLR(0.5),
        RandomAffine3D(rot=10., scale=(0.90, 1.10), translate=5, p=0.5),
        RandomElasticDeform(alpha=8.0, sigma=4.0, p=0.3),
        RandomBiasField(coeff=0.3, p=0.3),
        RandomNoise(std=(0., 0.04), p=0.4),
        RandomIntensityShift(shift=0.1, scale=(0.90, 1.10), p=0.4),
        RandomGamma(gamma=(0.80, 1.20), p=0.3),
        ClampNormalize(),
    ])

def get_val_augmentation() -> Compose3D:
    return Compose3D([ClampNormalize()])

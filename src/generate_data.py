"""
Synthetic polycrystalline microstructure generator.
Creates Voronoi-based grain structures for different alloy classes,
used as training data for the conditional generative model.
"""
import os
import numpy as np
from scipy.spatial import cKDTree
from PIL import Image

# Alloy "recipes": grain count controls average grain size, jitter controls
# how irregular the grain boundaries look.
ALLOYS = {
    0: {"name": "fine_steel", "n_grains": (400, 600), "jitter": 0.3},
    1: {"name": "medium_brass", "n_grains": (150, 250), "jitter": 0.5},
    2: {"name": "coarse_aluminum", "n_grains": (40, 80), "jitter": 0.8},
}
SIZE = 128
N_PER_CLASS = 1000
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def make_structure(n_grains, jitter, size=SIZE, rng=None):
    rng = rng or np.random.default_rng()
    rand = rng.random((n_grains, 2)) * size
    g = int(np.sqrt(n_grains)) + 1
    gx, gy = np.meshgrid(np.linspace(0, size, g), np.linspace(0, size, g))
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1)[:n_grains]
    if len(grid) < n_grains:
        grid = np.vstack([grid, rand[: n_grains - len(grid)]])
    seeds = (1 - jitter) * grid + jitter * rand

    yy, xx = np.mgrid[0:size, 0:size]
    pixels = np.stack([xx.ravel(), yy.ravel()], axis=1)
    _, grain_id = cKDTree(seeds).query(pixels)
    grain_id = grain_id.reshape(size, size)

    gray = rng.uniform(60, 220, n_grains)[grain_id]

    edge = np.zeros_like(grain_id, dtype=bool)
    edge[:-1, :] |= grain_id[:-1, :] != grain_id[1:, :]
    edge[:, :-1] |= grain_id[:, :-1] != grain_id[:, 1:]
    gray[edge] = 20

    gray += rng.normal(0, 4, gray.shape)
    return np.clip(gray, 0, 255).astype(np.uint8)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(42)
    images, labels = [], []

    for label, cfg in ALLOYS.items():
        for _ in range(N_PER_CLASS):
            n = rng.integers(*cfg["n_grains"])
            images.append(make_structure(n, cfg["jitter"], rng=rng))
            labels.append(label)

    images = np.array(images)
    labels = np.array(labels)
    np.savez_compressed(os.path.join(OUT_DIR, "microstructures.npz"),
                         images=images, labels=labels)

    for label, cfg in ALLOYS.items():
        idx = int(np.where(labels == label)[0][0])
        Image.fromarray(images[idx]).save(
            os.path.join(OUT_DIR, f"preview_{cfg['name']}.png"))

    print(f"Generated {len(images)} images across {len(ALLOYS)} alloy classes.")
    print(f"Saved to {OUT_DIR}/microstructures.npz")


if __name__ == "__main__":
    main()

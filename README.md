# Alloy Microstructure Generator

A conditional generative model that creates synthetic polycrystalline
grain-structure images for different alloy types (fine steel, medium
brass, coarse aluminum — easy to extend to more classes).

## Approach

1. **Synthetic data generation** (`src/generate_data.py`): since no real
   micrograph dataset was available, grain structures are generated using
   Voronoi tessellation, with grain count and boundary jitter tuned per
   alloy class to approximate real microstructural differences (grain
   size, boundary irregularity).
2. **Model** (`src/model.py`): a Conditional VAE (cVAE) that takes the
   alloy label as a condition (one-hot, concatenated into the encoder and
   decoder) and learns to generate grain-structure images matching that
   class.
3. **Training** (`src/train.py`): standard VAE ELBO loss (reconstruction +
   KL divergence), with sample grids saved every 5 epochs to track
   progress.
4. **Sampling** (`src/sample.py`): generates new, unseen microstructures
   for each alloy class from the trained model.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# 1. Generate the synthetic training data
python src/generate_data.py

# 2. Train the conditional VAE
python src/train.py

# 3. Generate new samples per alloy from the trained model
python src/sample.py
```

Outputs (sample grids per epoch, final generated images per alloy, and
the trained model checkpoint) are saved to `outputs/`.

## Project structure

```
microstructure-gen/
├── data/               # synthetic dataset + preview images
├── src/
│   ├── generate_data.py  # Voronoi-based synthetic data generator
│   ├── model.py           # Conditional VAE architecture
│   ├── train.py           # training loop
│   └── sample.py          # generate new samples from trained model
├── notebooks/          # optional Colab/Jupyter notebook version
├── outputs/             # generated samples + checkpoint (gitignored)
└── requirements.txt
```

## Limitations

- Trained on **synthetic** Voronoi-based data, not real alloy
  micrographs — this is a proof of concept for the generative pipeline,
  not a materials-accurate simulator.
- Built as a rapid (1-day) project; further work could include training
  on real EBSD/micrograph datasets, using a diffusion model for higher
  fidelity, and adding more alloy classes with real compositional
  conditioning (not just class labels).

## Author

Built as a course/personal project exploring conditional generative
models for materials microstructure synthesis.

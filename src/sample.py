"""
Generate new microstructure samples from a trained model, per alloy class.
Usage: python src/sample.py
"""
import os
import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from model import ConditionalVAE
from generate_data import ALLOYS

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
CKPT = os.path.join(OUT_DIR, "cvae_microstructure.pt")
N_CLASSES = 3
SAMPLES_PER_CLASS = 8


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConditionalVAE(n_classes=N_CLASSES).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device))
    model.eval()

    with torch.no_grad():
        for label, cfg in ALLOYS.items():
            y_onehot = F.one_hot(
                torch.full((SAMPLES_PER_CLASS,), label), N_CLASSES
            ).float().to(device)
            samples = model.sample(y_onehot, device)
            path = os.path.join(OUT_DIR, f"generated_{cfg['name']}.png")
            save_image(samples, path, nrow=4)
            print(f"Saved {path}")


if __name__ == "__main__":
    main()

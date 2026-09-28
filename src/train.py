"""
Trains the conditional VAE on the synthetic microstructure dataset.
Usage: python src/train.py
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision.utils import save_image

from model import ConditionalVAE, vae_loss

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "microstructures.npz")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
N_CLASSES = 3
EPOCHS = 30
BATCH_SIZE = 32
LR = 1e-3


class MicrostructureDataset(Dataset):
    def __init__(self, path):
        d = np.load(path)
        self.images = d["images"].astype(np.float32) / 255.0
        self.labels = d["labels"].astype(np.int64)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = torch.from_numpy(self.images[idx]).unsqueeze(0)  # (1,H,W)
        label = torch.tensor(self.labels[idx])
        return img, label


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT_DIR, exist_ok=True)

    dataset = MicrostructureDataset(DATA_PATH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = ConditionalVAE(n_classes=N_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for img, label in loader:
            img = img.to(device)
            y_onehot = F.one_hot(label, N_CLASSES).float().to(device)

            optimizer.zero_grad()
            recon, mu, logvar = model(img, y_onehot)
            loss, recon_l, kld = vae_loss(recon, img, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * img.size(0)

        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch:02d}/{EPOCHS}  loss={avg_loss:.2f}")

        if epoch % 5 == 0 or epoch == EPOCHS:
            model.eval()
            with torch.no_grad():
                labels = torch.arange(N_CLASSES).repeat_interleave(4)
                y_onehot = F.one_hot(labels, N_CLASSES).float().to(device)
                samples = model.sample(y_onehot, device)
                save_image(samples, os.path.join(OUT_DIR, f"samples_epoch{epoch}.png"),
                           nrow=4)

    torch.save(model.state_dict(), os.path.join(OUT_DIR, "cvae_microstructure.pt"))
    print("Training complete. Model and samples saved to outputs/")


if __name__ == "__main__":
    main()

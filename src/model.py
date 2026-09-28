"""
Conditional VAE for generating alloy microstructures.
Condition = alloy class label (one-hot), so the model learns to generate
grain structures matching a requested alloy type.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalVAE(nn.Module):
    def __init__(self, img_size=128, n_classes=3, latent_dim=64):
        super().__init__()
        self.img_size = img_size
        self.n_classes = n_classes
        self.latent_dim = latent_dim

        # Encoder: image + class -> latent distribution
        self.enc_conv = nn.Sequential(
            nn.Conv2d(1 + n_classes, 32, 4, 2, 1), nn.ReLU(),   # 64
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),              # 32
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),             # 16
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(),            # 8
        )
        self.fc_mu = nn.Linear(256 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(256 * 8 * 8, latent_dim)

        # Decoder: latent + class -> image
        self.fc_dec = nn.Linear(latent_dim + n_classes, 256 * 8 * 8)
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(),   # 16
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),    # 32
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),     # 64
            nn.ConvTranspose2d(32, 1, 4, 2, 1), nn.Sigmoid(),   # 128
        )

    def encode(self, x, y_onehot):
        y_map = y_onehot.view(y_onehot.size(0), self.n_classes, 1, 1)
        y_map = y_map.expand(-1, -1, self.img_size, self.img_size)
        h = self.enc_conv(torch.cat([x, y_map], dim=1))
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z, y_onehot):
        h = self.fc_dec(torch.cat([z, y_onehot], dim=1))
        h = h.view(-1, 256, 8, 8)
        return self.dec_conv(h)

    def forward(self, x, y_onehot):
        mu, logvar = self.encode(x, y_onehot)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, y_onehot)
        return recon, mu, logvar

    def sample(self, y_onehot, device):
        z = torch.randn(y_onehot.size(0), self.latent_dim, device=device)
        return self.decode(z, y_onehot)


def vae_loss(recon, x, mu, logvar, kld_weight=0.001):
    recon_loss = F.binary_cross_entropy(recon, x, reduction="sum") / x.size(0)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon_loss + kld_weight * kld, recon_loss, kld

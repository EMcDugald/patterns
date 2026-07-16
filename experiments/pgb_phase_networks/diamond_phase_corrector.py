import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

class PhaseCorrectorDataset(torch.utils.data.Dataset):
    """
    Simple dataset wrapping a single probe_fields.npz from diamond_phase_data_probe.

    For now this is a single-sample dataset; __len__ returns 1 and the
    DataLoader just reuses the same sample across epochs.
    """

    def __init__(self, probe_npz_path):
        d = np.load(probe_npz_path, allow_pickle=True)

        u = d["u"]          # (Ny, Nx)
        A = d["A"]          # (Ny, Nx)
        phase = d["phase"]  # (Ny, Nx)
        mask = d["valid_mask"]    # (Ny, Nx)
        ramp_n = d["ramp_n"]      # (Ny, Nx)

        # Exclude any locations where phase is NaN from the valid region
        phase_nan = np.isnan(phase)
        if phase_nan.any():
            print("Phase has NaNs; tightening mask to exclude them.")
            mask = np.where(phase_nan, 0.0, mask)

        print("effective valid fraction after phase-NaN cleanup:", float(np.mean(mask)))

        # Replace NaNs in phase so all downstream tensor ops stay finite
        phase = np.where(phase_nan, 0.0, phase)

        self.u = torch.tensor(u, dtype=torch.float32)
        self.A = torch.tensor(A, dtype=torch.float32)
        self.phase = torch.tensor(phase, dtype=torch.float32)
        self.mask = torch.tensor(mask, dtype=torch.float32)
        self.ramp_n = torch.tensor(ramp_n, dtype=torch.float32)

        cos_phase = torch.cos(self.phase)
        sin_phase = torch.sin(self.phase)

        # feature stack: [u, A, cos(theta), sin(theta), ramp_n]
        feats = torch.stack(
            [self.u, self.A, cos_phase, sin_phase, self.ramp_n],
            dim=0
        )  # (C, Ny, Nx)

        # single sample dataset for now
        self.feats = feats.unsqueeze(0)        # (1, C, Ny, Nx)
        self.u_true = self.u.unsqueeze(0)      # (1, Ny, Nx)
        self.A_true = self.A.unsqueeze(0)      # (1, Ny, Nx)
        self.phase_true = self.phase.unsqueeze(0)  # (1, Ny, Nx)
        self.mask_valid = self.mask.unsqueeze(0)   # (1, Ny, Nx)

        print("u nan/inf:",
              torch.isnan(self.u).any().item(),
              torch.isinf(self.u).any().item())
        print("A nan/inf:",
              torch.isnan(self.A).any().item(),
              torch.isinf(self.A).any().item())
        print("phase nan/inf:",
              torch.isnan(self.phase).any().item(),
              torch.isinf(self.phase).any().item())
        print("mask valid fraction:", float(self.mask.mean()))
        print("ramp_n min/max:", float(self.ramp_n.min()), float(self.ramp_n.max()))

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return (
            self.feats[idx],       # (C, Ny, Nx)
            self.u_true[idx],      # (Ny, Nx)
            self.A_true[idx],      # (Ny, Nx)
            self.phase_true[idx],  # (Ny, Nx)
            self.mask_valid[idx],  # (Ny, Nx)
        )


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

class ResidualPhaseCorrector(nn.Module):
    """
    Small CNN that takes the feature stack and predicts delta_theta on the grid.
    """

    def __init__(self, in_channels=5, hidden_channels=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        # x: (B, C, Ny, Nx)
        delta_theta = self.net(x)  # (B, 1, Ny, Nx)
        return delta_theta


# ---------------------------------------------------------------------
# Losses / regularizers
# ---------------------------------------------------------------------

def masked_mse(u_pred, u_true, mask):
    """
    Mean squared error computed only on mask==1 locations.

    u_pred, u_true, mask: (B, Ny, Nx)
    """
    diff2 = (u_pred - u_true) ** 2 * mask
    num = torch.sum(mask)
    return torch.sum(diff2) / torch.clamp(num, min=1.0)


def phase_regularizers(delta_theta, mask):
    """
    Simple correction regularizers:

    - smallness: ||delta_theta||^2 on valid points
    - smoothness: ||∇ delta_theta||^2 on valid points
    """
    dt = delta_theta.squeeze(1)  # (B, Ny, Nx)

    reg_small = torch.mean((dt ** 2) * mask)

    dy = dt[:, 1:, :] - dt[:, :-1, :]
    dx = dt[:, :, 1:] - dt[:, :, :-1]
    mask_dy = mask[:, 1:, :]
    mask_dx = mask[:, :, 1:]

    reg_smooth = (
        torch.mean((dy ** 2) * mask_dy) +
        torch.mean((dx ** 2) * mask_dx)
    )

    return reg_small, reg_smooth


# ---------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------

def train_phase_corrector(
    probe_npz_path,
    out_dir,
    num_epochs=200,
    lr=1e-3,
    lam_small=1e-3,
    lam_smooth=1e-3,
):
    ds = PhaseCorrectorDataset(probe_npz_path)
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResidualPhaseCorrector(in_channels=5, hidden_channels=32).to(device)
    optim_ = optim.Adam(model.parameters(), lr=lr)

    history = []

    for epoch in range(num_epochs):
        model.train()
        for feats, u_true, A_true, phase_true, mask_valid in loader:
            feats = feats.to(device)         # (B, C, Ny, Nx)
            u_true = u_true.to(device)       # (B, Ny, Nx)
            A_true = A_true.to(device)       # (B, Ny, Nx)
            phase_true = phase_true.to(device)  # (B, Ny, Nx)
            mask_valid = mask_valid.to(device)  # (B, Ny, Nx)

            delta_theta = model(feats)  # (B, 1, Ny, Nx)

            # phi_tot = theta + delta_theta
            phi_tot = phase_true.unsqueeze(1) + delta_theta  # (B, 1, Ny, Nx)
            u_pred = A_true.unsqueeze(1) * torch.cos(phi_tot)
            u_pred = u_pred.squeeze(1)  # (B, Ny, Nx)

            loss_rec = masked_mse(u_pred, u_true, mask_valid)
            reg_small, reg_smooth = phase_regularizers(delta_theta, mask_valid)
            loss = loss_rec + lam_small * reg_small + lam_smooth * reg_smooth

            optim_.zero_grad()
            loss.backward()
            optim_.step()

        history.append(
            (
                float(loss.item()),
                float(loss_rec.item()),
                float(reg_small.item()),
                float(reg_smooth.item()),
            )
        )

        if (epoch + 1) % 1 == 0:
            print(
                f"Epoch {epoch+1}/{num_epochs}  "
                f"loss={loss.item():.5e}  "
                f"rec={loss_rec.item():.5e}  "
                f"small={reg_small.item():.5e}  "
                f"smooth={reg_smooth.item():.5e}"
            )

    # save model & history
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), out_dir / "phase_corrector.pt")

    hist_arr = np.array(history, dtype=float)
    np.savez_compressed(
        out_dir / "training_history.npz",
        history=hist_arr,
        columns=np.array(["loss", "loss_rec", "reg_small", "reg_smooth"]),
    )

    # final evaluation on the single sample, for plotting
    model.eval()
    with torch.no_grad():
        feats, u_true, A_true, phase_true, mask_valid = next(iter(loader))
        feats = feats.to(device)
        u_true = u_true.to(device)
        A_true = A_true.to(device)
        phase_true = phase_true.to(device)
        mask_valid = mask_valid.to(device)

        delta_theta = model(feats)  # (1,1,Ny,Nx)
        phi_tot = phase_true.unsqueeze(1) + delta_theta
        u_pred = A_true.unsqueeze(1) * torch.cos(phi_tot)
        u_pred = u_pred.squeeze(1)  # (1,Ny,Nx)

    # back to numpy for plotting
    u_true_np = u_true.squeeze(0).cpu().numpy()
    A_true_np = A_true.squeeze(0).cpu().numpy()
    phase_true_np = phase_true.squeeze(0).cpu().numpy()
    mask_np = mask_valid.squeeze(0).cpu().numpy()
    delta_theta_np = delta_theta.squeeze(0).squeeze(0).cpu().numpy()
    phi_tot_np = phi_tot.squeeze(0).squeeze(0).cpu().numpy()
    u_pred_np = u_pred.squeeze(0).cpu().numpy()

    fig, axs = plt.subplots(2, 3, figsize=(13, 8))
    axs = axs.ravel()

    im = axs[0].imshow(u_true_np, cmap="copper")
    axs[0].set_title("u_true")
    plt.colorbar(im, ax=axs[0], shrink=0.8)

    im = axs[1].imshow(A_true_np * np.cos(phase_true_np), cmap="copper")
    axs[1].set_title("A cos(theta) (baseline)")
    plt.colorbar(im, ax=axs[1], shrink=0.8)

    im = axs[2].imshow(u_pred_np, cmap="copper")
    axs[2].set_title("A cos(theta + delta) (corrected)")
    plt.colorbar(im, ax=axs[2], shrink=0.8)

    im = axs[3].imshow(phase_true_np, cmap="twilight")
    axs[3].set_title("theta (baseline)")
    plt.colorbar(im, ax=axs[3], shrink=0.8)

    im = axs[4].imshow(delta_theta_np * mask_np, cmap="twilight")
    axs[4].set_title("delta_theta * mask")
    plt.colorbar(im, ax=axs[4], shrink=0.8)

    im = axs[5].imshow(phi_tot_np * mask_np, cmap="twilight")
    axs[5].set_title("phi_tot * mask")
    plt.colorbar(im, ax=axs[5], shrink=0.8)

    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    fig.savefig(out_dir / "phase_corrector_comparison.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

def main(args=None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--probe_npz",
        type=str,
        required=True,
        help="Path to probe_fields.npz produced by diamond_phase_data_probe.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for model and figures.",
    )
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lam_small", type=float, default=1e-3)
    parser.add_argument("--lam_smooth", type=float, default=1e-3)

    ns = parser.parse_args(args)

    if ns.out_dir is not None:
        out_dir = ns.out_dir
    else:
        out_dir = _HERE / "results" / "diamond_phase_corrector"

    train_phase_corrector(
        ns.probe_npz,
        out_dir,
        num_epochs=ns.num_epochs,
        lr=ns.lr,
        lam_small=ns.lam_small,
        lam_smooth=ns.lam_smooth,
    )


if __name__ == "__main__":
    if len(sys.argv) == 1:

        class _Args:
            probe_npz = (
                "/Users/edwardmcdugald/patterns/experiments/pgb_phase_networks/"
                "results/diamond_phase_probe/"
                "sh_pgb_diamond_mu0.450_T50_N2_Ny1536_Ly188.496_margin0.45_"
                "knee_uhu_sigma1.571_dm0.400_ts0.40_gsnone_phase_diamond_"
                "ns192_ds0.125_bdinner_ksym_prmsaved_sif0.200_srm0.990_rst0.000/"
                "data/probe_fields.npz"
            )
            out_dir = (
                "/Users/edwardmcdugald/patterns/experiments/pgb_phase_networks/"
                "results/diamond_phase_corrector/"
                "sh_pgb_diamond_mu0.450_T50_N2_Ny1536_Ly188.496_margin0.45_"
                "knee_uhu_sigma1.571_dm0.400_ts0.40"
            )
            num_epochs = 20
            lr = 1e-3
            lam_small = 1e-8
            lam_smooth = 1e-8

        a = _Args()
        main(
            [
                "--probe_npz",
                a.probe_npz,
                "--out_dir",
                a.out_dir,
                "--num_epochs",
                str(a.num_epochs),
                "--lr",
                str(a.lr),
                "--lam_small",
                str(a.lam_small),
                "--lam_smooth",
                str(a.lam_smooth),
            ]
        )
    else:
        main()
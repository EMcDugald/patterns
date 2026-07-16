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

from numpy.fft import fft2, ifft2, fftfreq


_HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------------
# Spectral macro filter
# ---------------------------------------------------------------------

def macro(arr, sigma, Lx, Ly):
    """
    Gaussian spectral smoothing on a periodic rectangle.
    Matches original calc_order_parameter_and_phase_sh_rectangle.py.
    """
    M, N = np.shape(arr)
    a = 0.5 * Lx
    b = 0.5 * Ly
    kx = (np.pi / a) * fftfreq(N, d=1 / N)
    ky = (np.pi / b) * fftfreq(M, d=1 / M)
    xi, eta = np.meshgrid(kx, ky)
    kernel = np.exp(-0.5 * sigma**2 * (xi**2 + eta**2))
    return np.real(ifft2(kernel * fft2(arr)))


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _choose_key(d, candidates):
    for k in candidates:
        if k in d:
            return k
    return None


def _norm01(arr):
    arr = np.asarray(arr, dtype=float)
    amin = np.nanmin(arr)
    amax = np.nanmax(arr)
    return (arr - amin) / (amax - amin + 1e-12)


def _extract_2d(arr):
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        return arr[..., -1]
    raise ValueError(f"Expected 2D or 3D array, got shape {arr.shape}")


def _load_raw_micro_energy(raw_npz_path):
    d = np.load(raw_npz_path, allow_pickle=True)

    energy_key = _choose_key(
        d,
        ["e"]
    )
    if energy_key is None:
        raise KeyError(
            "Could not find a usable micro-energy field in raw file. "
            "Tried keys: 'u', 'uu', 'micro_energy', 'micro_u', 'pattern', 'pattern_raw'."
        )

    micro = _extract_2d(d[energy_key])

    ramp_key = _choose_key(d, ["phase_ramp", "ramp_inner", "ramp"])
    ramp = _extract_2d(d[ramp_key]) if ramp_key is not None else None

    if "x" not in d or "y" not in d:
        raise KeyError("Expected x and y arrays in raw npz file.")

    x = np.asarray(d["x"]).ravel()
    y = np.asarray(d["y"]).ravel()
    Lx = float(x[-1] - x[0])
    Ly = float(y[-1] - y[0])

    return {
        "micro": micro,
        "micro_key": energy_key,
        "ramp": ramp,
        "ramp_key": ramp_key,
        "x": x,
        "y": y,
        "Lx": Lx,
        "Ly": Ly,
    }


def build_macro_weight_map(raw_npz_path, sigma, use_raw_ramp=True, weight_floor=0.0):
    info = _load_raw_micro_energy(raw_npz_path)

    micro_used = np.array(info["micro"], dtype=float, copy=True)
    if use_raw_ramp and info["ramp"] is not None:
        ramp_n = _norm01(info["ramp"])
        micro_used = micro_used * ramp_n

    macro_energy = macro(micro_used, sigma=sigma, Lx=info["Lx"], Ly=info["Ly"])
    weights = np.abs(macro_energy)
    weights = _norm01(weights)

    if weight_floor > 0.0:
        weights = weight_floor + (1.0 - weight_floor) * weights

    info["micro_used"] = micro_used
    info["macro_energy"] = macro_energy
    info["weights"] = weights
    return info


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

class WeightedPhaseCorrectorDataset(torch.utils.data.Dataset):
    """
    Single-sample dataset:
      - probe_fields.npz provides u, A, phase, valid_mask, ramp_n
      - raw SH npz provides micro energy -> macro energy -> weight map
    """

    def __init__(
        self,
        probe_npz_path,
        raw_npz_path,
        macro_sigma,
        use_raw_ramp=True,
        weight_floor=0.0,
        strict_ramp_thresh=None,
    ):
        d = np.load(probe_npz_path, allow_pickle=True)

        u = np.asarray(d["u"], dtype=float)
        A = np.asarray(d["A"], dtype=float)
        phase = np.asarray(d["phase"], dtype=float)
        mask = np.asarray(d["valid_mask"], dtype=float)
        ramp_n = np.asarray(d["ramp_n"], dtype=float)

        phase_nan = np.isnan(phase)
        if phase_nan.any():
            print("Phase has NaNs; tightening mask to exclude them.")
            mask = np.where(phase_nan, 0.0, mask)

        phase = np.where(phase_nan, 0.0, phase)

        if strict_ramp_thresh is not None:
            strict_mask = (ramp_n >= strict_ramp_thresh).astype(float)
            mask = mask * strict_mask
            print(f"Applied stricter ramp threshold: {strict_ramp_thresh}")

        print("effective valid fraction after cleanup:", float(np.mean(mask)))

        macro_info = build_macro_weight_map(
            raw_npz_path,
            sigma=macro_sigma,
            use_raw_ramp=use_raw_ramp,
            weight_floor=weight_floor,
        )
        weights = np.asarray(macro_info["weights"], dtype=float)

        if weights.shape != u.shape:
            raise ValueError(
                f"Weight map shape {weights.shape} does not match probe field shape {u.shape}"
            )

        self.u = torch.tensor(u, dtype=torch.float32)
        self.A = torch.tensor(A, dtype=torch.float32)
        self.phase = torch.tensor(phase, dtype=torch.float32)
        self.mask = torch.tensor(mask, dtype=torch.float32)
        self.ramp_n = torch.tensor(ramp_n, dtype=torch.float32)
        self.weights = torch.tensor(weights, dtype=torch.float32)

        cos_phase = torch.cos(self.phase)
        sin_phase = torch.sin(self.phase)

        # feature stack: [u, A, cos(theta), sin(theta), ramp_n, weights]
        feats = torch.stack(
            [self.u, self.A, cos_phase, sin_phase, self.ramp_n, self.weights],
            dim=0
        )

        self.feats = feats.unsqueeze(0)             # (1, C, Ny, Nx)
        self.u_true = self.u.unsqueeze(0)           # (1, Ny, Nx)
        self.A_true = self.A.unsqueeze(0)           # (1, Ny, Nx)
        self.phase_true = self.phase.unsqueeze(0)   # (1, Ny, Nx)
        self.mask_valid = self.mask.unsqueeze(0)    # (1, Ny, Nx)
        self.weight_map = self.weights.unsqueeze(0) # (1, Ny, Nx)

        self.macro_info = macro_info

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
        print("weight map min/max:", float(self.weights.min()), float(self.weights.max()))
        print("raw micro key:", macro_info["micro_key"])
        print("raw ramp key:", macro_info["ramp_key"])

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return (
            self.feats[idx],
            self.u_true[idx],
            self.A_true[idx],
            self.phase_true[idx],
            self.mask_valid[idx],
            self.weight_map[idx],
        )


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

class ResidualPhaseCorrector(nn.Module):
    """
    Small CNN that predicts delta_theta on the grid.
    """

    def __init__(self, in_channels=6, hidden_channels=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------
# Losses / regularizers
# ---------------------------------------------------------------------

def weighted_masked_mse(u_pred, u_true, mask, weights):
    """
    Weighted MSE on the valid region.
    """
    effective_w = mask * weights
    diff2 = (u_pred - u_true) ** 2
    denom = torch.sum(effective_w)
    return torch.sum(effective_w * diff2) / torch.clamp(denom, min=1.0)


def phase_regularizers(delta_theta, mask):
    """
    Smallness + smoothness regularization on valid points.
    """
    dt = delta_theta.squeeze(1)

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

def train_phase_corrector_weighted(
    probe_npz_path,
    raw_npz_path,
    out_dir,
    macro_sigma,
    num_epochs=200,
    lr=1e-3,
    lam_small=1e-8,
    lam_smooth=1e-8,
    use_raw_ramp=True,
    weight_floor=0.0,
    strict_ramp_thresh=None,
):
    ds = WeightedPhaseCorrectorDataset(
        probe_npz_path=probe_npz_path,
        raw_npz_path=raw_npz_path,
        macro_sigma=macro_sigma,
        use_raw_ramp=use_raw_ramp,
        weight_floor=weight_floor,
        strict_ramp_thresh=strict_ramp_thresh,
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResidualPhaseCorrector(in_channels=6, hidden_channels=32).to(device)
    optim_ = optim.Adam(model.parameters(), lr=lr)

    history = []

    for epoch in range(num_epochs):
        model.train()
        for feats, u_true, A_true, phase_true, mask_valid, weight_map in loader:
            feats = feats.to(device)
            u_true = u_true.to(device)
            A_true = A_true.to(device)
            phase_true = phase_true.to(device)
            mask_valid = mask_valid.to(device)
            weight_map = weight_map.to(device)

            delta_theta = model(feats)

            phi_tot = phase_true.unsqueeze(1) + delta_theta
            u_pred = A_true.unsqueeze(1) * torch.cos(phi_tot)
            u_pred = u_pred.squeeze(1)

            loss_rec = weighted_masked_mse(u_pred, u_true, mask_valid, weight_map)
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

        print(
            f"Epoch {epoch+1}/{num_epochs}  "
            f"loss={loss.item():.5e}  "
            f"rec={loss_rec.item():.5e}  "
            f"small={reg_small.item():.5e}  "
            f"smooth={reg_smooth.item():.5e}"
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), out_dir / "phase_corrector_macro_weighted.pt")

    hist_arr = np.array(history, dtype=float)
    np.savez_compressed(
        out_dir / "training_history.npz",
        history=hist_arr,
        columns=np.array(["loss", "loss_rec", "reg_small", "reg_smooth"]),
    )

    model.eval()
    with torch.no_grad():
        feats, u_true, A_true, phase_true, mask_valid, weight_map = next(iter(loader))
        feats = feats.to(device)
        u_true = u_true.to(device)
        A_true = A_true.to(device)
        phase_true = phase_true.to(device)
        mask_valid = mask_valid.to(device)
        weight_map = weight_map.to(device)

        delta_theta = model(feats)
        phi_tot = phase_true.unsqueeze(1) + delta_theta
        u_pred = A_true.unsqueeze(1) * torch.cos(phi_tot)
        u_pred = u_pred.squeeze(1)

    u_true_np = u_true.squeeze(0).cpu().numpy()
    A_true_np = A_true.squeeze(0).cpu().numpy()
    phase_true_np = phase_true.squeeze(0).cpu().numpy()
    mask_np = mask_valid.squeeze(0).cpu().numpy()
    weight_np = weight_map.squeeze(0).cpu().numpy()
    delta_theta_np = delta_theta.squeeze(0).squeeze(0).cpu().numpy()
    phi_tot_np = phi_tot.squeeze(0).squeeze(0).cpu().numpy()
    u_pred_np = u_pred.squeeze(0).cpu().numpy()

    macro_info = ds.macro_info

    Ny, Nx = phase_true_np.shape
    iy_mid = Ny // 2
    ix0 = Nx // 2

    y_line = np.arange(Ny)
    phase_line = phase_true_np[:, ix0]
    phi_tot_line = phi_tot_np[:, ix0]
    mask_line = mask_np[:, ix0] > 0.5

    fig, axs = plt.subplots(4, 3, figsize=(14, 14))
    axs = axs.ravel()

    im = axs[0].imshow(u_true_np, cmap="copper")
    axs[0].set_title("u_true")
    plt.colorbar(im, ax=axs[0], shrink=0.8)

    im = axs[1].imshow(A_true_np * np.cos(phase_true_np), cmap="copper")
    axs[1].set_title("A cos(theta) baseline")
    plt.colorbar(im, ax=axs[1], shrink=0.8)

    im = axs[2].imshow(u_pred_np, cmap="copper")
    axs[2].set_title("A cos(theta + delta) corrected")
    plt.colorbar(im, ax=axs[2], shrink=0.8)

    im = axs[3].imshow(phase_true_np * mask_np, cmap="twilight")
    axs[3].set_title("theta * mask")
    plt.colorbar(im, ax=axs[3], shrink=0.8)

    im = axs[4].imshow(delta_theta_np * mask_np, cmap="twilight")
    axs[4].set_title("delta_theta * mask")
    plt.colorbar(im, ax=axs[4], shrink=0.8)

    im = axs[5].imshow(phi_tot_np * mask_np, cmap="twilight")
    axs[5].set_title("phi_tot * mask")
    plt.colorbar(im, ax=axs[5], shrink=0.8)

    im = axs[6].imshow(np.abs(macro_info["micro_used"]), cmap="viridis")
    axs[6].set_title("|micro energy used|")
    plt.colorbar(im, ax=axs[6], shrink=0.8)

    im = axs[7].imshow(np.abs(macro_info["macro_energy"]), cmap="viridis")
    axs[7].set_title("|macro energy|")
    plt.colorbar(im, ax=axs[7], shrink=0.8)

    im = axs[8].imshow(weight_np * mask_np, cmap="magma")
    axs[8].set_title("weight map * mask")
    plt.colorbar(im, ax=axs[8], shrink=0.8)

    axs[9].plot(y_line[mask_line], phase_line[mask_line], lw=2, label="baseline phase")
    axs[9].plot(y_line[mask_line], phi_tot_line[mask_line], lw=2, label="corrected phase")
    axs[9].set_title("Phase along x=0 column")
    axs[9].set_xlabel("y-index")
    axs[9].set_ylabel("phase")
    axs[9].legend()

    axs[10].plot(y_line[mask_line], (phi_tot_line - phase_line)[mask_line], lw=2, color="tab:red")
    axs[10].set_title("Delta phase along x=0 column")
    axs[10].set_xlabel("y-index")
    axs[10].set_ylabel("delta_theta")

    axs[11].axis("off")
    axs[11].text(
        0.0, 1.0,
        f"raw micro key: {macro_info['micro_key']}\n"
        f"raw ramp key: {macro_info['ramp_key']}\n"
        f"valid fraction: {mask_np.mean():.4f}\n"
        f"weight min/max: {weight_np.min():.4f}, {weight_np.max():.4f}",
        va="top",
        ha="left",
        family="monospace",
        fontsize=10,
    )

    for ax in axs[:9]:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    fig.savefig(out_dir / "phase_corrector_macro_weighted_comparison.png", dpi=200)
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
        "--raw_npz",
        type=str,
        required=True,
        help="Path to raw SH npz containing micro energy and optional ramp.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for model and figures.",
    )
    parser.add_argument("--macro_sigma", type=float, default=1.0)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lam_small", type=float, default=1e-8)
    parser.add_argument("--lam_smooth", type=float, default=1e-8)
    parser.add_argument("--weight_floor", type=float, default=0.0)
    parser.add_argument("--strict_ramp_thresh", type=float, default=None)
    parser.add_argument(
        "--no_raw_ramp",
        action="store_true",
        help="If set, do not multiply raw micro energy by raw ramp before macro filtering.",
    )

    ns = parser.parse_args(args)

    if ns.out_dir is not None:
        out_dir = ns.out_dir
    else:
        out_dir = _HERE / "results" / "diamond_phase_corrector_macro_weighted"

    train_phase_corrector_weighted(
        probe_npz_path=ns.probe_npz,
        raw_npz_path=ns.raw_npz,
        out_dir=out_dir,
        macro_sigma=ns.macro_sigma,
        num_epochs=ns.num_epochs,
        lr=ns.lr,
        lam_small=ns.lam_small,
        lam_smooth=ns.lam_smooth,
        use_raw_ramp=not ns.no_raw_ramp,
        weight_floor=ns.weight_floor,
        strict_ramp_thresh=ns.strict_ramp_thresh,
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
            raw_npz = (
                "/Users/edwardmcdugald/patterns/pipelines/data/sh_pgb_diamond/"
                "debug/raw/sh_pgb_diamond_mu0.450_T50_N2_Ny1536_Ly188.496_"
                "margin0.45_knee.npz"
            )
            out_dir = (
                "/Users/edwardmcdugald/patterns/experiments/pgb_phase_networks/"
                "results/diamond_phase_corrector_macro_weighted/"
                "sh_pgb_diamond_mu0.450_T50_N2_Ny1536_Ly188.496_margin0.45_"
                "knee_uhu_sigma1.571_dm0.400_ts0.40"
            )
            macro_sigma = np.pi/2
            num_epochs = 20
            lr = 1e-3
            lam_small = 1e-8
            lam_smooth = 1e-8
            weight_floor = 0.0
            strict_ramp_thresh = None
            no_raw_ramp = False

        a = _Args()
        argv = [
            "--probe_npz", a.probe_npz,
            "--raw_npz", a.raw_npz,
            "--out_dir", a.out_dir,
            "--macro_sigma", str(a.macro_sigma),
            "--num_epochs", str(a.num_epochs),
            "--lr", str(a.lr),
            "--lam_small", str(a.lam_small),
            "--lam_smooth", str(a.lam_smooth),
            "--weight_floor", str(a.weight_floor),
        ]
        if a.strict_ramp_thresh is not None:
            argv += ["--strict_ramp_thresh", str(a.strict_ramp_thresh)]
        if a.no_raw_ramp:
            argv.append("--no_raw_ramp")
        main(argv)
    else:
        main()
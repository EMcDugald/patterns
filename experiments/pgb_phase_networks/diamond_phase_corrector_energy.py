import argparse
import json
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


def _norm01(arr):
    arr = np.asarray(arr, dtype=float)
    amin = np.nanmin(arr)
    amax = np.nanmax(arr)
    return (arr - amin) / (amax - amin + 1e-12)


def _as_frame_stack(arr, name, Ny=None, Nx=None):
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr[None, ...]
    if arr.ndim != 3:
        raise ValueError(f"{name}: expected 2D or 3D array, got shape {arr.shape}")
    if Ny is not None and Nx is not None:
        if arr.shape[1] == Ny and arr.shape[2] == Nx:
            return arr
        if arr.shape[0] == Ny and arr.shape[1] == Nx:
            return np.transpose(arr, (2, 0, 1))
    if arr.shape[0] <= 16:
        return arr
    if arr.shape[-1] <= 16:
        return np.transpose(arr, (2, 0, 1))
    raise ValueError(
        f"{name}: could not infer frame axis for shape {arr.shape}. "
        "Please transpose explicitly."
    )


def _get_first_middle_last_indices(T):
    idx = [0, T // 2, T - 1]
    out = []
    for i in idx:
        if i not in out:
            out.append(i)
    return out


def _frame_tag(frame_idx, num_frames):
    if frame_idx == 0:
        return "first"
    if frame_idx == num_frames - 1:
        return "last"
    if frame_idx == num_frames // 2:
        return "middle"
    return f"frame{frame_idx:03d}"


def _fmt_float(x):
    s = f"{float(x):.3g}"
    return s.replace("-", "m").replace(".", "p")


def build_run_name(
    probe_npz_path,
    use_macro_weighting,
    use_sym_phase,
    weight_floor,
    strict_ramp_thresh,
    lr,
    lam_small,
    lam_smooth,
    num_epochs,
):
    probe_path = Path(probe_npz_path).resolve()
    try:
        stem = probe_path.parents[1].name
    except Exception:
        stem = probe_path.stem
    parts = [
        stem,
        "symphase" if use_sym_phase else "rawphase",
        "mw1" if use_macro_weighting else "mw0",
        f"wf{_fmt_float(weight_floor)}",
        f"ep{num_epochs}",
        f"lr{_fmt_float(lr)}",
        f"ls{_fmt_float(lam_small)}",
        f"lm{_fmt_float(lam_smooth)}",
    ]
    if strict_ramp_thresh is not None:
        parts.append(f"rt{_fmt_float(strict_ramp_thresh)}")
    return "_".join(parts)


def build_weight_map_from_probe(macro_energy, use_macro_weighting=True, weight_floor=0.0):
    macro_energy = np.asarray(macro_energy, dtype=float)
    if use_macro_weighting:
        weights = np.empty_like(macro_energy, dtype=float)
        for t in range(macro_energy.shape[0]):
            weights[t] = _norm01(np.abs(macro_energy[t]))
            if weight_floor > 0.0:
                weights[t] = weight_floor + (1.0 - weight_floor) * weights[t]
    else:
        weights = np.ones_like(macro_energy, dtype=float)
    return weights


class WeightedPhaseCorrectorDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        probe_npz_path,
        use_macro_weighting=True,
        weight_floor=0.0,
        strict_ramp_thresh=None,
        use_sym_phase=True,
    ):
        d = np.load(probe_npz_path, allow_pickle=True)
        if "x" not in d or "y" not in d:
            raise KeyError("Probe file must contain x and y arrays.")
        self.x = np.asarray(d["x"], dtype=float).ravel()
        self.y = np.asarray(d["y"], dtype=float).ravel()
        Ny = len(self.y)
        Nx = len(self.x)

        u = _as_frame_stack(np.asarray(d["u"], dtype=float), "u", Ny=Ny, Nx=Nx)
        # A = _as_frame_stack(np.asarray(d["A"], dtype=float), "A", Ny=Ny, Nx=Nx)
        # simple constant-amplitude experiment:
        # one scalar per frame = max(|u|) over that frame
        A_const = np.abs(u).max(axis=(1, 2), keepdims=True)
        A = np.broadcast_to(A_const, u.shape).copy()

        phase_key = "phase_sym" if (use_sym_phase and "phase_sym" in d) else "phase"
        if phase_key not in d:
            raise KeyError("Probe file must contain phase_sym or phase.")
        phase = _as_frame_stack(np.asarray(d[phase_key], dtype=float), phase_key, Ny=Ny, Nx=Nx)
        mask = _as_frame_stack(np.asarray(d["valid_mask"], dtype=float), "valid_mask", Ny=Ny, Nx=Nx)
        ramp_n = _as_frame_stack(np.asarray(d["ramp_n"], dtype=float), "ramp_n", Ny=Ny, Nx=Nx)
        micro_energy = _as_frame_stack(np.asarray(d["micro_energy"], dtype=float), "micro_energy", Ny=Ny, Nx=Nx)
        macro_energy = _as_frame_stack(np.asarray(d["macro_energy"], dtype=float), "macro_energy", Ny=Ny, Nx=Nx)

        T, Ny_u, Nx_u = u.shape
        expected_shape = (T, Ny_u, Nx_u)
        for name, arr in [
            ("A", A),
            (phase_key, phase),
            ("valid_mask", mask),
            ("ramp_n", ramp_n),
            ("micro_energy", micro_energy),
            ("macro_energy", macro_energy),
        ]:
            if arr.shape != expected_shape:
                raise ValueError(f"{name} shape {arr.shape} does not match u shape {expected_shape}.")

        phase_nan = np.isnan(phase)
        if phase_nan.any():
            print("Phase has NaNs; tightening mask to exclude them.")
            mask = np.where(phase_nan, 0.0, mask)
        phase = np.where(phase_nan, 0.0, phase)

        if strict_ramp_thresh is not None:
            strict_mask = (ramp_n >= strict_ramp_thresh).astype(float)
            mask = mask * strict_mask
            print(f"Applied stricter ramp threshold: {strict_ramp_thresh}")

        weights = build_weight_map_from_probe(
            macro_energy,
            use_macro_weighting=use_macro_weighting,
            weight_floor=weight_floor,
        )

        self.u = torch.tensor(u, dtype=torch.float32)
        self.A = torch.tensor(A, dtype=torch.float32)
        self.phase = torch.tensor(phase, dtype=torch.float32)
        self.mask = torch.tensor(mask, dtype=torch.float32)
        self.ramp_n = torch.tensor(ramp_n, dtype=torch.float32)
        self.micro_energy = torch.tensor(micro_energy, dtype=torch.float32)
        self.macro_energy = torch.tensor(macro_energy, dtype=torch.float32)
        self.weights = torch.tensor(weights, dtype=torch.float32)

        cos_phase = torch.cos(self.phase)
        sin_phase = torch.sin(self.phase)
        self.feats = torch.stack(
            [self.u, self.A, cos_phase, sin_phase, self.ramp_n, self.weights], dim=1
        )

        self.u_true = self.u
        self.A_true = self.A
        self.phase_true = self.phase
        self.mask_valid = self.mask
        self.weight_map = 1000*self.weights
        self.phase_key = phase_key
        self.use_macro_weighting = bool(use_macro_weighting)
        self.num_frames = T
        self.plot_frame_ids = _get_first_middle_last_indices(T)
        self.info = {
            "phase_key": phase_key,
            "use_macro_weighting": bool(use_macro_weighting),
            "weight_floor": float(weight_floor),
            "strict_ramp_thresh": None if strict_ramp_thresh is None else float(strict_ramp_thresh),
        }

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx):
        return (
            self.feats[idx],
            self.u_true[idx],
            self.A_true[idx],
            self.phase_true[idx],
            self.mask_valid[idx],
            self.weight_map[idx],
            idx,
        )


class ResidualPhaseCorrector(nn.Module):
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


def weighted_masked_mse(u_pred, u_true, mask, weights):
    effective_w = mask * weights
    diff2 = (u_pred - u_true) ** 2
    denom = torch.sum(effective_w)
    return torch.sum(effective_w * diff2) / torch.clamp(denom, min=1.0)


def phase_regularizers(delta_theta, mask):
    dt = delta_theta.squeeze(1)
    reg_small = torch.mean((dt ** 2) * mask)
    dy = dt[:, 1:, :] - dt[:, :-1, :]
    dx = dt[:, :, 1:] - dt[:, :, :-1]
    mask_dy = mask[:, 1:, :]
    mask_dx = mask[:, :, 1:]
    reg_smooth = torch.mean((dy ** 2) * mask_dy) + torch.mean((dx ** 2) * mask_dx)
    return reg_small, reg_smooth


def _plot_frame_summary(
    out_dir,
    frame_idx,
    num_frames,
    x,
    y,
    u_true_np,
    A_true_np,
    phase_true_np,
    mask_np,
    weight_np,
    delta_theta_np,
    phi_tot_np,
    u_pred_np,
    micro_energy_np,
    macro_energy_np,
    phase_key,
    use_macro_weighting,
):
    Ny, Nx = phase_true_np.shape
    ix0 = Nx // 2
    iy0 = Ny // 2
    y_line = y
    x_line = x

    phase_vertical = phase_true_np[:, ix0]
    phi_tot_vertical = phi_tot_np[:, ix0]
    mask_vertical = mask_np[:, ix0] > 0.5
    phase_horizontal = phase_true_np[iy0, :]
    phi_tot_horizontal = phi_tot_np[iy0, :]
    mask_horizontal = mask_np[iy0, :] > 0.5

    fig, axs = plt.subplots(4, 4, figsize=(18, 14))
    axs = axs.ravel()

    im = axs[0].imshow(u_true_np, cmap="copper")
    axs[0].set_title("u_true")
    plt.colorbar(im, ax=axs[0], shrink=0.8)

    im = axs[1].imshow((A_true_np * np.cos(phase_true_np)) * mask_np, cmap="copper")
    axs[1].set_title("A cos(theta) * mask")
    plt.colorbar(im, ax=axs[1], shrink=0.8)

    im = axs[2].imshow(u_pred_np * mask_np, cmap="copper")
    axs[2].set_title("A cos(theta + delta) * mask")
    plt.colorbar(im, ax=axs[2], shrink=0.8)

    im = axs[3].imshow(mask_np, cmap="gray")
    axs[3].set_title("valid mask")
    plt.colorbar(im, ax=axs[3], shrink=0.8)

    im = axs[4].imshow(phase_true_np * mask_np, cmap="twilight")
    axs[4].set_title(f"{phase_key} * mask")
    plt.colorbar(im, ax=axs[4], shrink=0.8)

    im = axs[5].imshow(delta_theta_np * mask_np, cmap="twilight")
    axs[5].set_title("delta_theta * mask")
    plt.colorbar(im, ax=axs[5], shrink=0.8)

    im = axs[6].imshow(phi_tot_np * mask_np, cmap="twilight")
    axs[6].set_title("phi_tot * mask")
    plt.colorbar(im, ax=axs[6], shrink=0.8)

    im = axs[7].imshow(weight_np * mask_np, cmap="magma")
    axs[7].set_title("weight map * mask")
    plt.colorbar(im, ax=axs[7], shrink=0.8)

    im = axs[8].imshow(np.abs(micro_energy_np) * mask_np, cmap="viridis")
    axs[8].set_title("|micro energy| * mask")
    plt.colorbar(im, ax=axs[8], shrink=0.8)

    im = axs[9].imshow(macro_energy_np * mask_np, cmap="coolwarm")
    axs[9].set_title("macro energy * mask")
    plt.colorbar(im, ax=axs[9], shrink=0.8)

    im = axs[10].imshow(np.abs(macro_energy_np) * mask_np, cmap="viridis")
    axs[10].set_title("|macro energy| * mask")
    plt.colorbar(im, ax=axs[10], shrink=0.8)

    axs[11].plot(y_line[mask_vertical], phase_vertical[mask_vertical], lw=2, label="baseline phase")
    axs[11].plot(y_line[mask_vertical], phi_tot_vertical[mask_vertical], lw=2, label="corrected phase")
    axs[11].set_title("Phase along vertical center line")
    axs[11].set_xlabel("y")
    axs[11].set_ylabel("phase")
    axs[11].legend()

    axs[12].plot(y_line[mask_vertical], (phi_tot_vertical - phase_vertical)[mask_vertical], lw=2, color="tab:red")
    axs[12].set_title("Delta phase along vertical center line")
    axs[12].set_xlabel("y")
    axs[12].set_ylabel("delta_theta")

    axs[13].plot(x_line[mask_horizontal], phase_horizontal[mask_horizontal], lw=2, label="baseline phase")
    axs[13].plot(x_line[mask_horizontal], phi_tot_horizontal[mask_horizontal], lw=2, label="corrected phase")
    axs[13].set_title("Phase along horizontal center line (x-axis cut)")
    axs[13].set_xlabel("x")
    axs[13].set_ylabel("phase")
    axs[13].legend()

    axs[14].plot(x_line[mask_horizontal], (phi_tot_horizontal - phase_horizontal)[mask_horizontal], lw=2, color="tab:red")
    axs[14].set_title("Delta phase along horizontal center line (x-axis cut)")
    axs[14].set_xlabel("x")
    axs[14].set_ylabel("delta_theta")

    axs[15].axis("off")
    axs[15].text(
        0.0, 1.0,
        f"frame: {frame_idx} / {num_frames - 1}\n"
        f"phase key: {phase_key}\n"
        f"use macro weighting: {use_macro_weighting}\n"
        f"valid fraction: {mask_np.mean():.4f}\n"
        f"weight min/max: {weight_np.min():.4f}, {weight_np.max():.4f}",
        va="top", ha="left", family="monospace", fontsize=10,
    )

    for ax in axs[:11]:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    tag = _frame_tag(frame_idx, num_frames)
    fig.savefig(out_dir / f"phase_corrector_comparison_{tag}.png", dpi=200)
    plt.close(fig)


def train_phase_corrector_weighted(
    probe_npz_path,
    out_dir,
    num_epochs=200,
    lr=1e-3,
    lam_small=1e-8,
    lam_smooth=1e-8,
    use_macro_weighting=True,
    weight_floor=0.0,
    strict_ramp_thresh=None,
    use_sym_phase=True,
):
    ds = WeightedPhaseCorrectorDataset(
        probe_npz_path=probe_npz_path,
        use_macro_weighting=use_macro_weighting,
        weight_floor=weight_floor,
        strict_ramp_thresh=strict_ramp_thresh,
        use_sym_phase=use_sym_phase,
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResidualPhaseCorrector(in_channels=6, hidden_channels=32).to(device)
    optim_ = optim.Adam(model.parameters(), lr=lr)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params} trainable parameters.")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    history = []
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_rec = 0.0
        epoch_small = 0.0
        epoch_smooth = 0.0
        nb = 0

        for feats, u_true, A_true, phase_true, mask_valid, weight_map, frame_idx in loader:
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

            epoch_loss += float(loss.item())
            epoch_rec += float(loss_rec.item())
            epoch_small += float(reg_small.item())
            epoch_smooth += float(reg_smooth.item())
            nb += 1

        history.append((
            epoch_loss / max(nb, 1),
            epoch_rec / max(nb, 1),
            epoch_small / max(nb, 1),
            epoch_smooth / max(nb, 1),
        ))
        print(
            f"Epoch {epoch+1}/{num_epochs}  "
            f"loss={history[-1][0]:.5e}  "
            f"rec={history[-1][1]:.5e}  "
            f"small={history[-1][2]:.5e}  "
            f"smooth={history[-1][3]:.5e}"
        )

    torch.save(model.state_dict(), out_dir / "phase_corrector.pt")
    hist_arr = np.array(history, dtype=float)
    np.savez_compressed(
        out_dir / "training_history.npz",
        history=hist_arr,
        columns=np.array(["loss", "loss_rec", "reg_small", "reg_smooth"]),
    )

    run_config = {
        "probe_npz_path": str(probe_npz_path),
        "out_dir": str(out_dir),
        "num_epochs": int(num_epochs),
        "lr": float(lr),
        "lam_small": float(lam_small),
        "lam_smooth": float(lam_smooth),
        "use_macro_weighting": bool(use_macro_weighting),
        "weight_floor": float(weight_floor),
        "strict_ramp_thresh": None if strict_ramp_thresh is None else float(strict_ramp_thresh),
        "use_sym_phase": bool(use_sym_phase),
        "dataset_info": ds.info,
        "plot_frame_ids": [int(i) for i in ds.plot_frame_ids],
    }
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    model.eval()
    with torch.no_grad():
        for frame_idx in ds.plot_frame_ids:
            feats, u_true, A_true, phase_true, mask_valid, weight_map, _ = ds[frame_idx]
            feats = feats.unsqueeze(0).to(device)
            u_true = u_true.unsqueeze(0).to(device)
            A_true = A_true.unsqueeze(0).to(device)
            phase_true = phase_true.unsqueeze(0).to(device)
            mask_valid = mask_valid.unsqueeze(0).to(device)
            weight_map = weight_map.unsqueeze(0).to(device)

            delta_theta = model(feats)
            phi_tot = phase_true.unsqueeze(1) + delta_theta
            u_pred = A_true.unsqueeze(1) * torch.cos(phi_tot)
            u_pred = u_pred.squeeze(1)

            _plot_frame_summary(
                out_dir=out_dir,
                frame_idx=frame_idx,
                num_frames=ds.num_frames,
                x=ds.x,
                y=ds.y,
                u_true_np=u_true.squeeze(0).cpu().numpy(),
                A_true_np=A_true.squeeze(0).cpu().numpy(),
                phase_true_np=phase_true.squeeze(0).cpu().numpy(),
                mask_np=mask_valid.squeeze(0).cpu().numpy(),
                weight_np=weight_map.squeeze(0).cpu().numpy(),
                delta_theta_np=delta_theta.squeeze(0).squeeze(0).cpu().numpy(),
                phi_tot_np=phi_tot.squeeze(0).squeeze(0).cpu().numpy(),
                u_pred_np=u_pred.squeeze(0).cpu().numpy(),
                micro_energy_np=ds.micro_energy[frame_idx].cpu().numpy(),
                macro_energy_np=ds.macro_energy[frame_idx].cpu().numpy(),
                phase_key=ds.phase_key,
                use_macro_weighting=ds.use_macro_weighting,
            )


def main(args=None):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--probe_npz", type=str, required=True, help="Path to probe_fields.npz produced by diamond_phase_data_probe.")
    parser.add_argument("--out_dir", type=str, default=None, help="Optional explicit output directory. If omitted, a parameter-coded directory is created.")
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lam_small", type=float, default=1e-8)
    parser.add_argument("--lam_smooth", type=float, default=1e-8)
    parser.add_argument("--weight_floor", type=float, default=0.0)
    parser.add_argument("--strict_ramp_thresh", type=float, default=None)
    parser.add_argument("--no_macro_weighting", action="store_true", help="If set, use uniform loss weights instead of normalized |macro_energy|.")
    parser.add_argument("--no_sym_phase", action="store_true", help="If set, use phase instead of phase_sym.")
    ns = parser.parse_args(args)

    use_macro_weighting = not ns.no_macro_weighting
    use_sym_phase = not ns.no_sym_phase

    if ns.out_dir is not None:
        out_dir = Path(ns.out_dir)
    else:
        run_name = build_run_name(
            probe_npz_path=ns.probe_npz,
            use_macro_weighting=use_macro_weighting,
            use_sym_phase=use_sym_phase,
            weight_floor=ns.weight_floor,
            strict_ramp_thresh=ns.strict_ramp_thresh,
            lr=ns.lr,
            lam_small=ns.lam_small,
            lam_smooth=ns.lam_smooth,
            num_epochs=ns.num_epochs,
        )
        out_dir = _HERE / "results" / "diamond_phase_corrector" / run_name

    train_phase_corrector_weighted(
        probe_npz_path=ns.probe_npz,
        out_dir=out_dir,
        num_epochs=ns.num_epochs,
        lr=ns.lr,
        lam_small=ns.lam_small,
        lam_smooth=ns.lam_smooth,
        use_macro_weighting=use_macro_weighting,
        weight_floor=ns.weight_floor,
        strict_ramp_thresh=ns.strict_ramp_thresh,
        use_sym_phase=use_sym_phase,
    )


if __name__ == "__main__":
    if len(sys.argv) == 1:
        class _Args:
            probe_npz = (
                "/Users/edwardmcdugald/patterns/experiments/pgb_phase_networks/results/diamond_phase_probe/sh_pgb_diamond_mu0.450_T50_N2_Ny1536_Ly188.496_margin0.45_knee_uhu_sigma1.571_dm0.400_ts0.40_gsnone_phase_diamond_ns192_ds0.125_bdinner_ksym_prmsaved_sif0.200_srm0.990_rst0.000/data/probe_fields.npz"
            )
            out_dir = None
            num_epochs = 20
            lr = 5e-4
            lam_small = 1e-8
            lam_smooth = 1.0
            weight_floor = 0.0
            strict_ramp_thresh = None
            no_macro_weighting = False
            no_sym_phase = False

        a = _Args()
        argv = [
            "--probe_npz", a.probe_npz,
            "--num_epochs", str(a.num_epochs),
            "--lr", str(a.lr),
            "--lam_small", str(a.lam_small),
            "--lam_smooth", str(a.lam_smooth),
            "--weight_floor", str(a.weight_floor),
        ]
        if a.out_dir is not None:
            argv += ["--out_dir", a.out_dir]
        if a.strict_ramp_thresh is not None:
            argv += ["--strict_ramp_thresh", str(a.strict_ramp_thresh)]
        if a.no_macro_weighting:
            argv.append("--no_macro_weighting")
        if a.no_sym_phase:
            argv.append("--no_sym_phase")
        main(argv)
    else:
        main()
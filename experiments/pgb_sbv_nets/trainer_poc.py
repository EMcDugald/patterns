# import json
# from pathlib import Path
#
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
# from torch.optim import Adam
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
#
#
# # ======================================================================
# # Dataset: one sample per (file, frame)
# # ======================================================================
#
# class SBVGridFramesDataset(Dataset):
#     """
#     Dataset for grid surrogate on all frames.
#
#     Each (file, frame k) is a separate sample.
#
#     - Reads probe_fields.npz + probe_summary.json.
#     - For each file, uses all T frames: k = 0..T-1.
#     - Builds input channels on Ny x Nx for each frame.
#     """
#
#     def __init__(self, root_dir, use_theta_init=False, include_singular=False):
#         """
#         root_dir: directory containing subdirs per stem with data/probe_fields.npz, data/probe_summary.json.
#         use_theta_init: if True, include theta_init_from_data channel for phase-correction tests.
#         include_singular: if True, also compute a singular energy target (placeholder).
#         """
#         self.root_dir = Path(root_dir)
#         self.use_theta_init = use_theta_init
#         self.include_singular = include_singular
#
#         # Collect (pf_path, ps_path, T) for all files
#         self.files = []
#         for stem_dir in sorted(self.root_dir.glob("*")):
#             data_dir = stem_dir / "data"
#             pf = data_dir / "probe_fields.npz"
#             ps = data_dir / "probe_summary.json"
#             if pf.is_file() and ps.is_file():
#                 dat = np.load(pf, allow_pickle=True)
#                 u = np.asarray(dat["u"])
#                 T = u.shape[2]
#                 self.files.append((pf, ps, T))
#
#         if not self.files:
#             raise RuntimeError(f"No probe_fields.npz found under {root_dir}")
#
#         # Build a flat index of (file_idx, frame_idx)
#         self.index = []
#         for fi, (_, _, T) in enumerate(self.files):
#             for k in range(T):
#                 self.index.append((fi, k))
#
#     def __len__(self):
#         return len(self.index)
#
#     def __getitem__(self, idx):
#         fi, k = self.index[idx]
#         pf_path, ps_path, T = self.files[fi]
#
#         dat = np.load(pf_path, allow_pickle=True)
#         with open(ps_path, "r") as f:
#             summary = json.load(f)
#
#         # Basic arrays
#         u = np.asarray(dat["u"])            # (Ny, Nx, T)
#         A = np.asarray(dat["A"])           # (Ny, Nx, T)
#         macro_energy = np.asarray(dat["macro_energy"])  # (Ny, Nx, T)
#         ramp_n = np.asarray(dat["ramp_n"])             # (Ny, Nx, T)
#         valid_mask = np.asarray(dat["valid_mask"]).astype(bool)  # (Ny, Nx, T)
#
#         # Optional fields
#         phase = dat["phase"] if "phase" in dat else None
#         theta_initial = dat["theta_initial"] if "theta_initial" in dat else None
#
#         # Build channels for this frame k
#         channels = []
#
#         channels.append(u[..., k])
#         channels.append(A[..., k])
#         channels.append(macro_energy[..., k])
#         channels.append(ramp_n[..., k])
#         channels.append(valid_mask[..., k].astype(float))
#
#         if phase is not None:
#             phase_k = np.asarray(phase)[..., k]
#             channels.append(phase_k)
#
#         if self.use_theta_init and theta_initial is not None:
#             # time-independent initial phase from raw SH data
#             theta_init_2d = np.asarray(theta_initial)  # (Ny, Nx)
#             channels.append(theta_init_2d)
#
#         arr = np.stack(channels, axis=0)  # (C, Ny, Nx)
#
#         # Bulk energy target from macro_energy and valid_mask at frame k (raw sum)
#         macro_k = macro_energy[..., k]
#         mask_k = valid_mask[..., k]
#         E_bulk = float((macro_k * mask_k).sum())
#
#         # Singular energy target (placeholder)
#         if self.include_singular:
#             # Example: singular energy could be norm of gradient of phase or
#             # some interface length measure. For now we use a dummy zero.
#             E_singular = 0.0
#         else:
#             E_singular = 0.0
#
#         mu = summary.get("mu", None)
#
#         x = torch.from_numpy(arr).float()     # (C, Ny, Nx)
#         y_bulk = torch.tensor([E_bulk], dtype=torch.float32)
#         y_singular = torch.tensor([E_singular], dtype=torch.float32)
#
#         meta = {
#             "pf_path": str(pf_path),
#             "stem": summary.get("stem", pf_path.parent.name),
#             "frame_index": int(k),
#             "mu": mu,
#         }
#
#         return x, y_bulk, y_singular, meta
#
#
# # ======================================================================
# # Models
# # ======================================================================
#
# class BulkEnergyGridCNN(nn.Module):
#     """
#     Simple CNN regressor on Ny x Nx grid for bulk energy.
#     """
#
#     def __init__(self, in_channels, hidden_channels=32):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(hidden_channels)
#         self.bn2 = nn.BatchNorm2d(hidden_channels)
#         self.bn3 = nn.BatchNorm2d(hidden_channels)
#
#         self.head = nn.Sequential(
#             nn.Linear(hidden_channels, hidden_channels),
#             nn.ReLU(inplace=True),
#             nn.Linear(hidden_channels, 1),
#         )
#
#     def forward(self, x):
#         # x: (B, C, Ny, Nx)
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.relu(self.bn3(self.conv3(x)))
#         # Global average pooling over Ny, Nx
#         x = x.mean(dim=(-2, -1))  # (B, hidden_channels)
#         out = self.head(x)        # (B, 1)
#         return out.squeeze(-1)    # (B,)
#
#
# class ThetaCorrectionGridCNN(nn.Module):
#     """
#     Model with:
#     - theta-correction branch producing delta_theta,
#     - bulk-energy branch that sees base channels + cos(theta_learned).
#     """
#
#     def __init__(self, base_in_channels, theta_in_channels=None, hidden_channels=32):
#         super().__init__()
#         if theta_in_channels is None:
#             theta_in_channels = base_in_channels
#
#         # Theta-correction branch
#         self.theta_conv1 = nn.Conv2d(theta_in_channels, hidden_channels, kernel_size=3, padding=1)
#         self.theta_conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
#         self.theta_out = nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1)  # delta_theta
#
#         # Bulk-energy branch: sees base channels + cos(theta_learned)
#         self.bulk_cnn = BulkEnergyGridCNN(in_channels=base_in_channels + 1,
#                                           hidden_channels=hidden_channels)
#
#     def forward(self, x, theta_init_from_data=None):
#         """
#         x: (B, C, Ny, Nx) base channels (including theta_init channel if you choose).
#         theta_init_from_data: (B, 1, Ny, Nx) or None.
#         """
#         B, C, Ny, Nx = x.shape
#
#         # If theta_init_from_data is not provided, assume last channel of x is theta_init
#         if theta_init_from_data is None:
#             theta_init = x[:, -1:, :, :]  # (B, 1, Ny, Nx)
#             theta_input = x
#         else:
#             theta_init = theta_init_from_data
#             theta_input = x
#
#         # Theta correction branch
#         t = F.relu(self.theta_conv1(theta_input))
#         t = F.relu(self.theta_conv2(t))
#         delta_theta = self.theta_out(t)          # (B, 1, Ny, Nx)
#         theta_learned = theta_init + delta_theta  # (B, 1, Ny, Nx)
#
#         # Cos(theta_learned) as extra channel
#         cos_theta = torch.cos(theta_learned)     # (B, 1, Ny, Nx)
#
#         # Concatenate cos(theta_learned) to base channels
#         x_aug = torch.cat([x, cos_theta], dim=1)  # (B, C+1, Ny, Nx)
#
#         # Bulk energy prediction
#         E_bulk_hat = self.bulk_cnn(x_aug)        # (B,)
#         return E_bulk_hat, theta_learned
#
#
# # ======================================================================
# # Plotting helpers
# # ======================================================================
#
# def plot_energy_scatter(model, dl, device, out_path):
#     model.eval()
#     ys = []
#     y_hats = []
#     with torch.no_grad():
#         for x, y_bulk, y_sing, meta in dl:
#             x = x.to(device)
#             y = y_bulk.to(device).view(-1)
#             y_hat = model(x)
#             ys.append(y.cpu().numpy())
#             y_hats.append(y_hat.cpu().numpy())
#     ys = np.concatenate(ys)
#     y_hats = np.concatenate(y_hats)
#
#     plt.figure()
#     plt.scatter(ys, y_hats, s=10, alpha=0.7)
#     plt.xlabel("True bulk energy")
#     plt.ylabel("Predicted bulk energy")
#     plt.title("Bulk energy: true vs predicted")
#     plt.grid(True)
#     out_path = Path(out_path)
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     plt.savefig(out_path, dpi=150)
#     plt.close()
#
#
# def plot_theta_and_energy(model, dl, device, out_dir):
#     out_dir = Path(out_dir)
#     out_dir.mkdir(parents=True, exist_ok=True)
#
#     model.eval()
#     with torch.no_grad():
#         # Take a few samples from the loader
#         for i, (x, y_bulk, y_sing, meta) in enumerate(dl):
#             if i >= 4:  # limit to a few examples
#                 break
#             x = x.to(device)
#             y = y_bulk.to(device).view(-1)
#             y_hat, theta_learned = model(x)
#
#             # Assume last channel of x is theta_init_from_data
#             theta_init = x[:, -1:, :, :]      # (B, 1, Ny, Nx)
#             cos_theta_init = torch.cos(theta_init)
#             cos_theta_learned = torch.cos(theta_learned)
#
#             # Use first sample in batch
#             cos_init_2d = cos_theta_init[0, 0].cpu().numpy()
#             cos_learned_2d = cos_theta_learned[0, 0].cpu().numpy()
#
#             # For convenience: assume channel 2 is macro_energy, channel 4 is valid_mask
#             macro_energy_2d = x[0, 2].cpu().numpy()
#             mask_2d = x[0, 4].cpu().numpy()
#
#             # Approximate jump set from theta_learned via gradient threshold
#             theta_learned_2d = theta_learned[0, 0].cpu().numpy()
#             grad_y, grad_x = np.gradient(theta_learned_2d)
#             grad_mag = np.sqrt(grad_x**2 + grad_y**2)
#             thresh = np.percentile(grad_mag, 95)
#             jump_mask = grad_mag > thresh
#
#             fig, axs = plt.subplots(2, 2, figsize=(10, 10))
#
#             im0 = axs[0, 0].imshow(cos_init_2d, cmap="gray", origin="lower")
#             axs[0, 0].set_title("cos(theta_init_from_data)")
#             plt.colorbar(im0, ax=axs[0, 0])
#
#             im1 = axs[0, 1].imshow(cos_learned_2d, cmap="gray", origin="lower")
#             axs[0, 1].set_title("cos(theta_learned)")
#             plt.colorbar(im1, ax=axs[0, 1])
#
#             im2 = axs[1, 0].imshow(macro_energy_2d * mask_2d, cmap="inferno", origin="lower")
#             axs[1, 0].set_title(f"macro_energy * mask\ntrue={y[0].item():.3f}, pred={y_hat[0].item():.3f}")
#             plt.colorbar(im2, ax=axs[1, 0])
#
#             im3 = axs[1, 1].imshow(jump_mask.astype(float), cmap="viridis", origin="lower")
#             axs[1, 1].set_title("approx jump set (grad threshold)")
#             plt.colorbar(im3, ax=axs[1, 1])
#
#             for ax in axs.ravel():
#                 ax.set_xticks([])
#                 ax.set_yticks([])
#
#             stem = meta[0]["stem"]
#             fig.suptitle(f"Sample {stem}, frame={meta[0]['frame_index']}", fontsize=11)
#             fig.tight_layout(rect=[0, 0, 1, 0.95])
#
#             fig.savefig(out_dir / f"{stem}_frame{meta[0]['frame_index']}_theta_energy.png", dpi=200)
#             plt.close(fig)
#
#
# # ======================================================================
# # Training functions for the 4 cases
# # ======================================================================
#
# def train_case(
#     data_root,
#     case_name,
#     use_theta_init,
#     include_singular,
#     epochs=50,
#     batch_size=4,
#     lr=1e-3,
#     device="cuda",
# ):
#     """
#     case_name: string for logging / file naming.
#     use_theta_init: bool (no-theta vs theta-correction).
#     include_singular: bool (bulk-only vs bulk+singular).
#
#     For now, singular energy is a placeholder; bulk-only is implemented.
#     """
#     ds = SBVGridFramesDataset(
#         root_dir=data_root,
#         use_theta_init=use_theta_init,
#         include_singular=include_singular,
#     )
#
#     # Quick sanity check
#     for i in range(min(5, len(ds))):
#         x_i, y_bulk_i, y_sing_i, meta_i = ds[i]
#         print("sample", i, "stem", meta_i["stem"], "frame", meta_i["frame_index"])
#         print("  E_bulk:", y_bulk_i.item())
#         print("  any NaN in macro_energy?", torch.isnan(x_i[2]).any().item())
#         print("  any NaN in valid_mask?", torch.isnan(x_i[4]).any().item())
#
#     dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
#
#     C = ds[0][0].shape[0]
#
#     if use_theta_init:
#         # case: theta correction
#         model = ThetaCorrectionGridCNN(base_in_channels=C).to(device)
#     else:
#         # case: no theta
#         model = BulkEnergyGridCNN(in_channels=C).to(device)
#
#     opt = Adam(model.parameters(), lr=lr)
#     loss_fn = nn.MSELoss()
#
#     for epoch in range(1, epochs + 1):
#         model.train()
#         total_loss = 0.0
#
#         for x, y_bulk, y_sing, meta in dl:
#             x = x.to(device)
#             y_bulk = y_bulk.to(device).view(-1)
#
#             opt.zero_grad()
#
#             if use_theta_init:
#                 y_hat_bulk, theta_learned = model(x)
#             else:
#                 y_hat_bulk = model(x)
#
#             # For now, loss is only on bulk energy
#             loss = loss_fn(y_hat_bulk, y_bulk)
#             loss.backward()
#             opt.step()
#
#             total_loss += loss.item() * x.size(0)
#
#         avg_loss = total_loss / len(ds)
#         print(f"[{case_name}] Epoch {epoch}: train MSE (bulk) = {avg_loss:.6f}")
#
#         # Diagnostics
#         if epoch % 1 == 0:
#             scatter_path = Path(data_root) / f"{case_name}_epoch{epoch}_scatter.png"
#             plot_energy_scatter(model if not use_theta_init else (lambda x: model(x)[0]),
#                                 dl, device, scatter_path)
#
#             if use_theta_init:
#                 plot_dir = Path(data_root) / f"{case_name}_epoch{epoch}_theta_plots"
#                 plot_theta_and_energy(model, dl, device, plot_dir)
#
#
# # ======================================================================
# # Convenience Args and main
# # ======================================================================
#
# if __name__ == "__main__":
#     class Args:
#         # root directory where probe outputs live:
#         # each OP stem has a subdir with data/probe_fields.npz, data/probe_summary.json
#         data_root = "/Users/edwardmcdugald/patterns/experiments/pgb_sbv_nets/results/sbv_phase_probe"
#
#         # Experiment toggle:
#         # 1: grid + bulk-only + no theta
#         # 2: grid + bulk-only + theta correction (theta_init_from_data)
#         # 3: grid + bulk+singular + no theta
#         # 4: grid + bulk+singular + theta correction
#         experiment_id = 1
#
#         # Training hyperparameters
#         epochs = 10
#         batch_size = 4
#         lr = 1e-3
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#
#     a = Args()
#
#     # Map experiment_id to settings
#     if a.experiment_id == 1:
#         case_name = "grid_bulk_only_no_theta"
#         use_theta_init = False
#         include_singular = False
#     elif a.experiment_id == 2:
#         case_name = "grid_bulk_only_theta_correction"
#         use_theta_init = True
#         include_singular = False
#     elif a.experiment_id == 3:
#         case_name = "grid_bulk_plus_singular_no_theta"
#         use_theta_init = False
#         include_singular = True
#     elif a.experiment_id == 4:
#         case_name = "grid_bulk_plus_singular_theta_correction"
#         use_theta_init = True
#         include_singular = True
#     else:
#         raise SystemExit(f"Unknown experiment_id={a.experiment_id} (expected 1–4).")
#
#     print(f"Running experiment: {case_name}")
#     train_case(
#         data_root=a.data_root,
#         case_name=case_name,
#         use_theta_init=use_theta_init,
#         include_singular=include_singular,
#         epochs=a.epochs,
#         batch_size=a.batch_size,
#         lr=a.lr,
#         device=a.device,
#     )




import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ======================================================================
# Dataset: one sample per (file, frame), energy density per cell
# ======================================================================

class SBVGridFramesDataset(Dataset):
    """
    Dataset for phase-based energy model on all frames.

    Each (file, frame k) is a separate sample.

    - Reads probe_fields.npz + probe_summary.json.
    - For each file, uses all T frames: k = 0..T-1.
    - Builds input channels on Ny x Nx for each frame.
    - Returns observed macro energy density and a valid mask for derivatives.
    """

    def __init__(self, root_dir, use_theta_init=False):
        """
        root_dir: directory containing subdirs per stem with data/probe_fields.npz, data/probe_summary.json.
        use_theta_init: if True, include theta_init_from_data channel.
        """
        self.root_dir = Path(root_dir)
        self.use_theta_init = use_theta_init

        # Collect (pf_path, ps_path, T) for all files
        self.files = []
        for stem_dir in sorted(self.root_dir.glob("*")):
            data_dir = stem_dir / "data"
            pf = data_dir / "probe_fields.npz"
            ps = data_dir / "probe_summary.json"
            if pf.is_file() and ps.is_file():
                dat = np.load(pf, allow_pickle=True)
                u = np.asarray(dat["u"])
                T = u.shape[2]
                self.files.append((pf, ps, T))

        if not self.files:
            raise RuntimeError(f"No probe_fields.npz found under {root_dir}")

        # Build a flat index of (file_idx, frame_idx)
        self.index = []
        for fi, (_, _, T) in enumerate(self.files):
            for k in range(T):
                self.index.append((fi, k))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        fi, k = self.index[idx]
        pf_path, ps_path, T = self.files[fi]

        dat = np.load(pf_path, allow_pickle=True)
        with open(ps_path, "r") as f:
            summary = json.load(f)

        # Basic arrays
        u = np.asarray(dat["u"])            # (Ny, Nx, T)
        A = np.asarray(dat["A"])           # (Ny, Nx, T)
        macro_energy = np.asarray(dat["macro_energy"])  # (Ny, Nx, T)
        ramp_n = np.asarray(dat["ramp_n"])             # (Ny, Nx, T)
        valid_mask = np.asarray(dat["valid_mask"]).astype(bool)  # (Ny, Nx, T)

        # Optional fields
        phase = dat["phase"] if "phase" in dat else None
        theta_initial = dat["theta_initial"] if "theta_initial" in dat else None

        # Frame k
        u_k = u[..., k]
        A_k = A[..., k]
        macro_k = macro_energy[..., k]
        ramp_k = ramp_n[..., k]
        valid_k = valid_mask[..., k]

        channels = []
        channels.append(u_k)
        channels.append(A_k)
        channels.append(macro_k)
        channels.append(ramp_k.astype(float))
        channels.append(valid_k.astype(float))

        # Optional phase channel from data
        if phase is not None:
            phase_k = np.asarray(phase)[..., k]
            channels.append(phase_k)

        # Optional theta_init_from_data channel
        if self.use_theta_init and theta_initial is not None:
            theta_init_2d = np.asarray(theta_initial)  # (Ny, Nx)
            channels.append(theta_init_2d)

        arr = np.stack(channels, axis=0)  # (C, Ny, Nx)

        # Observed macro energy density, masked
        E_macro_obs = macro_k * valid_k.astype(float)

        # Valid mask for derivatives: interior and not in ramp
        # You can refine this rule; here we avoid ramp_n != 0 and invalid cells.
        valid_deriv = (valid_k) & (ramp_k == 0)
        valid_deriv = valid_deriv.astype(float)

        # Torch tensors
        x = torch.from_numpy(arr).float()                    # (C, Ny, Nx)
        y_macro = torch.from_numpy(E_macro_obs).float().unsqueeze(0)      # (1, Ny, Nx)
        valid_mask_t = torch.from_numpy(valid_deriv).float().unsqueeze(0) # (1, Ny, Nx)

        mu = summary.get("mu", None)

        meta = {
            "pf_path": str(pf_path),
            "stem": summary.get("stem", pf_path.parent.name),
            "frame_index": int(k),
            "mu": mu,
        }

        return x, y_macro, valid_mask_t, meta


# ======================================================================
# Phase + singular energy model (E1/E3-style)
# ======================================================================

class PhaseSingularEnergyModel(nn.Module):
    """
    Phase-based energy model:

    E_bulk(x) = lambda_delta (Delta theta)^2 + lambda_g (1 - |grad theta|^2)^2
    E_sing(x) = lambda_J |sin(theta)| s(x)
    E_total(x) = E_bulk(x) + E_sing(x)
    """

    def __init__(self, in_channels, hidden_channels=32,
                 use_theta_init=False, use_singular=False,
                 lambda_delta=1.0, lambda_g=1.0, lambda_J=1.0):
        super().__init__()
        self.use_theta_init = use_theta_init
        self.use_singular = use_singular
        self.lambda_delta = lambda_delta
        self.lambda_g = lambda_g
        self.lambda_J = lambda_J

        # Base feature extractor on grid
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Phase head: predict delta_theta or theta directly
        self.phase_head = nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1)  # (B,1,Ny,Nx)

        # Singular head: predict s(x) >= 0
        if use_singular:
            self.s_head = nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1)

    def forward(self, x, theta_init=None, valid_mask=None):
        """
        x: (B, C, Ny, Nx) input channels.
        theta_init: (B,1,Ny,Nx) or None.
        valid_mask: (B,1,Ny,Nx) or None; used to mask derivatives.
        """
        B, C, Ny, Nx = x.shape

        feat = self.features(x)  # (B, hidden, Ny, Nx)

        delta_theta = self.phase_head(feat)  # (B,1,Ny,Nx)

        if self.use_theta_init and theta_init is not None:
            theta = theta_init + delta_theta
        else:
            theta = delta_theta  # learn theta directly

        if self.use_singular:
            s_raw = self.s_head(feat)
            s = F.softplus(s_raw)  # nonnegative s(x)
        else:
            s = None

        # Compute gradients and Laplacian with masking
        grad_sq, lap_sq = self._grad_and_lap_sq(theta, valid_mask)

        # Bulk energy density
        E_bulk = self.lambda_delta * lap_sq + self.lambda_g * (1.0 - grad_sq) ** 2

        # Singular energy density
        if self.use_singular and s is not None:
            E_sing = self.lambda_J * torch.abs(torch.sin(theta)) * s
            # Mask singular term where derivatives are invalid as well
            if valid_mask is not None:
                E_sing = E_sing * valid_mask
        else:
            E_sing = torch.zeros_like(E_bulk)

        E_total = E_bulk + E_sing  # (B,1,Ny,Nx)

        return {
            "theta": theta,
            "s": s,
            "E_bulk": E_bulk,
            "E_sing": E_sing,
            "E_total": E_total,
        }

    def _grad_and_lap_sq(self, theta, valid_mask):
        """
        Compute |grad theta|^2 and (lap theta)^2 on grid.
        Use central differences where possible, forward/backward at boundaries.
        Mask out ramp/invalid cells via valid_mask.
        """
        # theta: (B,1,Ny,Nx)
        B, _, Ny, Nx = theta.shape

        # d/dy
        dy = torch.zeros_like(theta)
        dy[:, :, 1:-1, :] = 0.5 * (theta[:, :, 2:, :] - theta[:, :, :-2, :])
        dy[:, :, 0, :] = theta[:, :, 1, :] - theta[:, :, 0, :]
        dy[:, :, -1, :] = theta[:, :, -1, :] - theta[:, :, -2, :]

        # d/dx
        dx = torch.zeros_like(theta)
        dx[:, :, :, 1:-1] = 0.5 * (theta[:, :, :, 2:] - theta[:, :, :, :-2])
        dx[:, :, :, 0] = theta[:, :, :, 1] - theta[:, :, :, 0]
        dx[:, :, :, -1] = theta[:, :, :, -1] - theta[:, :, :, -2]

        grad_sq = dx**2 + dy**2  # (B,1,Ny,Nx)

        # Laplacian via 5-point stencil (set zero at boundary)
        lap = torch.zeros_like(theta)
        lap[:, :, 1:-1, 1:-1] = (
            theta[:, :, 2:, 1:-1] + theta[:, :, :-2, 1:-1] +
            theta[:, :, 1:-1, 2:] + theta[:, :, 1:-1, :-2] -
            4 * theta[:, :, 1:-1, 1:-1]
        )
        lap_sq = lap**2

        # Mask out invalid / ramp-boundary cells
        if valid_mask is not None:
            grad_sq = grad_sq * valid_mask
            lap_sq = lap_sq * valid_mask

        return grad_sq, lap_sq


# ======================================================================
# Plotting helpers
# ======================================================================

def plot_energy_maps(model, dl, device, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    with torch.no_grad():
        # Take a few samples from the loader
        for i, (x, y_macro, valid_mask, meta) in enumerate(dl):
            if i >= 3:  # limit to a few examples
                break
            x = x.to(device)
            y_macro = y_macro.to(device)
            valid_mask = valid_mask.to(device)

            # theta_init_from_data as last channel if used
            if model.use_theta_init:
                theta_init = x[:, -1:, :, :]
            else:
                theta_init = None

            out = model(x, theta_init=theta_init, valid_mask=valid_mask)
            E_pred = out["E_total"]
            E_bulk = out["E_bulk"]
            E_sing = out["E_sing"]
            theta = out["theta"]

            # Use first sample
            E_obs_2d = y_macro[0, 0].cpu().numpy()
            E_pred_2d = E_pred[0, 0].cpu().numpy()
            E_bulk_2d = E_bulk[0, 0].cpu().numpy()
            E_sing_2d = E_sing[0, 0].cpu().numpy()
            theta_2d = theta[0, 0].cpu().numpy()

            fig, axs = plt.subplots(2, 3, figsize=(14, 8))

            im0 = axs[0, 0].imshow(E_obs_2d, cmap="inferno", origin="lower")
            axs[0, 0].set_title("Observed macro energy")
            plt.colorbar(im0, ax=axs[0, 0])

            im1 = axs[0, 1].imshow(E_pred_2d, cmap="inferno", origin="lower")
            axs[0, 1].set_title("Predicted total energy")
            plt.colorbar(im1, ax=axs[0, 1])

            im2 = axs[0, 2].imshow(E_bulk_2d, cmap="inferno", origin="lower")
            axs[0, 2].set_title("Predicted bulk energy")
            plt.colorbar(im2, ax=axs[0, 2])

            im3 = axs[1, 0].imshow(E_sing_2d, cmap="inferno", origin="lower")
            axs[1, 0].set_title("Predicted singular energy")
            plt.colorbar(im3, ax=axs[1, 0])

            im4 = axs[1, 1].imshow(theta_2d, cmap="twilight", origin="lower")
            axs[1, 1].set_title("Learned theta")
            plt.colorbar(im4, ax=axs[1, 1])

            im5 = axs[1, 2].imshow(valid_mask[0, 0].cpu().numpy(), cmap="gray", origin="lower")
            axs[1, 2].set_title("Valid derivative mask")
            plt.colorbar(im5, ax=axs[1, 2])

            for ax in axs.ravel():
                ax.set_xticks([])
                ax.set_yticks([])

            stem = meta["stem"]
            frame_idx = meta["frame_index"]
            fig.suptitle(f"Sample {stem}, frame={frame_idx}", fontsize=11)
            fig.tight_layout(rect=[0, 0, 1, 0.95])

            fig.savefig(out_dir / f"{stem}_frame{frame_idx}_energy_maps.png", dpi=200)
            plt.close(fig)


# ======================================================================
# Training function for the 4 cases (E1/E3 variants)
# ======================================================================

def train_case(
    data_root,
    case_name,
    use_theta_init,
    use_singular,
    lambda_delta=1.0,
    lambda_g=1.0,
    lambda_J=1.0,
    epochs=20,
    batch_size=1,
    lr=1e-3,
    device="cuda",
):
    ds = SBVGridFramesDataset(
        root_dir=data_root,
        use_theta_init=use_theta_init,
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    # Input channels count
    C = ds[0][0].shape[0]

    model = PhaseSingularEnergyModel(
        in_channels=C,
        hidden_channels=32,
        use_theta_init=use_theta_init,
        use_singular=use_singular,
        lambda_delta=lambda_delta,
        lambda_g=lambda_g,
        lambda_J=lambda_J,
    ).to(device)

    opt = Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0.0

        for x, y_macro, valid_mask, meta in dl:
            x = x.to(device)
            y_macro = y_macro.to(device)      # (B,1,Ny,Nx)
            valid_mask = valid_mask.to(device)

            opt.zero_grad()

            if model.use_theta_init:
                theta_init = x[:, -1:, :, :]
            else:
                theta_init = None

            out = model(x, theta_init=theta_init, valid_mask=valid_mask)
            E_pred = out["E_total"]

            # Masked L2 loss per cell
            diff = (E_pred - y_macro) * valid_mask
            loss = (diff**2).sum() / (valid_mask.sum() + 1e-8)

            loss.backward()
            opt.step()

            total_loss += loss.item()
            total_count += 1.0

        avg_loss = total_loss / max(total_count, 1.0)
        print(f"[{case_name}] Epoch {epoch}: masked MSE (density) = {avg_loss:.6e}")

        # Diagnostics
        if epoch % 5 == 0:
            plot_dir = Path(data_root) / f"{case_name}_epoch{epoch}_energy_plots"
            plot_energy_maps(model, dl, device, plot_dir)


# ======================================================================
# Convenience Args and main
# ======================================================================

if __name__ == "__main__":
    class Args:
        # root directory where probe outputs live:
        # each OP stem has a subdir with data/probe_fields.npz, data/probe_summary.json
        data_root = "/Users/edwardmcdugald/patterns/experiments/pgb_sbv_nets/results/sbv_phase_probe"

        # Experiment toggle:
        # 1: E1 bulk-only, no theta_init
        # 2: E1 bulk-only, with theta_init correction
        # 3: E3 bulk+singular, no theta_init (learn s)
        # 4: E3 bulk+singular, with theta_init correction
        experiment_id = 1

        # Energy weights
        lambda_delta = 1.0
        lambda_g = 1.0
        lambda_J = 1.0

        # Training hyperparameters
        epochs = 20
        batch_size = 1
        lr = 1e-3
        device = "cuda" if torch.cuda.is_available() else "cpu"

    a = Args()

    if a.experiment_id == 1:
        case_name = "E1_bulk_only_no_theta_init"
        use_theta_init = False
        use_singular = False
    elif a.experiment_id == 2:
        case_name = "E1_bulk_only_with_theta_init"
        use_theta_init = True
        use_singular = False
    elif a.experiment_id == 3:
        case_name = "E3_bulk_plus_singular_no_theta_init"
        use_theta_init = False
        use_singular = True
    elif a.experiment_id == 4:
        case_name = "E3_bulk_plus_singular_with_theta_init"
        use_theta_init = True
        use_singular = True
    else:
        raise SystemExit(f"Unknown experiment_id={a.experiment_id} (expected 1–4).")

    print(f"Running experiment: {case_name}")
    train_case(
        data_root=a.data_root,
        case_name=case_name,
        use_theta_init=use_theta_init,
        use_singular=use_singular,
        lambda_delta=a.lambda_delta,
        lambda_g=a.lambda_g,
        lambda_J=a.lambda_J,
        epochs=a.epochs,
        batch_size=a.batch_size,
        lr=a.lr,
        device=a.device,
    )
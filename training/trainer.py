"""Training and optimization loop for Single-Task and Multi-Task architectures on RTX 4050 / CUDA / CPU."""
import os
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import time
import csv
import json
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from training.losses import coral_to_probs, coral_predict, MultiTaskLoss, coral_loss
from evaluation.metrics import compute_fruit_metrics, compute_ripeness_metrics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_parameter_groups(model, model_type, lr, frozen=False):
    """Layer-wise learning rates: backbone 10x smaller than head."""
    if model_type == "multitask_resnet50":
        backbone_params = []
        head_params = []
        threshold_params = []
        for name, param in model.named_parameters():
            if "fruit_head" in name or "ripeness_fc" in name:
                head_params.append(param)
            elif "ripeness_bias" in name:
                threshold_params.append(param)
            else:
                param.requires_grad = not frozen
                backbone_params.append(param)

        groups = [
            {"params": head_params, "lr": lr},
            {"params": threshold_params, "lr": lr * 10.0, "weight_decay": 0.0}
        ]
        if not frozen:
            groups.append({"params": backbone_params, "lr": lr * 0.1})
        return groups

    elif model_type in ("resnet50_coral", "resnet18_coral"):
        head_p = []
        thresh_p = []
        body_p = []
        for name, param in model.named_parameters():
            if "fc.weight" in name or (hasattr(model, "fc") and name.startswith("fc.weight")):
                head_p.append(param)
            elif "bias" in name and param.dim() == 1:
                thresh_p.append(param)
            else:
                param.requires_grad = not frozen
                body_p.append(param)
        groups = [
            {"params": head_p, "lr": lr},
            {"params": thresh_p, "lr": lr * 10.0, "weight_decay": 0.0}
        ]
        if not frozen:
            groups.append({"params": body_p, "lr": lr * 0.1})
        return groups

    else:  # resnet50_softmax
        body_p = []
        head_p = []
        for name, param in model.named_parameters():
            if "fc" in name:
                head_p.append(param)
            else:
                param.requires_grad = not frozen
                body_p.append(param)
        groups = [{"params": head_p, "lr": lr}]
        if not frozen:
            groups.append({"params": body_p, "lr": lr * 0.1})
        return groups


class Trainer:
    def __init__(
        self,
        model,
        model_type,
        train_loader,
        val_loader,
        test_loader,
        exp_name,
        num_fruits=6,
        num_stages=4,
        lr=1e-4,
        weight_decay=1e-4,
        epochs=30,
        freeze_epochs=3,
        patience=8,
        lambda_ripeness=1.0,
        fruit_weights=None,
        ripeness_weights=None,
        grad_clip=1.0,
        device=None
    ):
        self.model = model
        self.model_type = model_type
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.exp_name = exp_name
        self.num_fruits = num_fruits
        self.num_stages = num_stages
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.freeze_epochs = freeze_epochs
        self.patience = patience
        self.lambda_ripeness = lambda_ripeness
        self.grad_clip = grad_clip

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.ckpt_dir = os.path.join(ROOT, "experiments", "checkpoints", exp_name)
        self.log_dir = os.path.join(ROOT, "experiments", "logs")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        self.loss_fn = MultiTaskLoss(
            num_fruits=num_fruits,
            num_stages=num_stages,
            lambda_ripeness=lambda_ripeness,
            fruit_weights=fruit_weights,
            ripeness_weights=ripeness_weights
        )

        self.use_amp = (self.device == "cuda")
        self.scaler = GradScaler("cuda", enabled=self.use_amp)

    def _forward_step(self, batch):
        images = batch["image"].to(self.device, non_blocking=True)
        fruit_id = batch["fruit_id"].to(self.device)
        ripeness_id = batch["ripeness_id"].to(self.device)
        has_fruit = batch["has_fruit"].to(self.device)
        has_ripeness = batch["has_ripeness"].to(self.device)

        if self.model_type == "multitask_resnet50":
            out = self.model(images)
            f_logits = out["fruit_logits"]
            r_logits = out["ripeness_logits"]
            tot_loss, f_loss, r_loss = self.loss_fn(
                f_logits, r_logits, fruit_id, ripeness_id, has_fruit, has_ripeness
            )
            return tot_loss, f_loss, r_loss, f_logits, r_logits

        elif self.model_type == "resnet50_softmax_fruit":
            f_logits = self.model(images)
            r_logits = None
            f_loss = nn.functional.cross_entropy(f_logits, fruit_id)
            return f_loss, f_loss, torch.tensor(0.0), f_logits, None

        elif self.model_type == "resnet50_softmax_ripeness":
            r_logits = self.model(images)
            r_loss = nn.functional.cross_entropy(r_logits, ripeness_id)
            return r_loss, torch.tensor(0.0), r_loss, None, r_logits

        elif self.model_type in ("resnet50_coral", "resnet18_coral"):
            r_logits = self.model(images)
            r_loss = coral_loss(r_logits, ripeness_id, num_stages=self.num_stages)
            return r_loss, torch.tensor(0.0), r_loss, None, r_logits

        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def train_epoch(self, optimizer):
        self.model.train()
        total_loss, total_f_loss, total_r_loss = 0.0, 0.0, 0.0
        n_samples = 0

        for batch in self.train_loader:
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda" if self.device == "cuda" else "cpu", enabled=self.use_amp):
                tot_loss, f_loss, r_loss, _, _ = self._forward_step(batch)

            self.scaler.scale(tot_loss).backward()
            if self.grad_clip > 0:
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.scaler.step(optimizer)
            self.scaler.update()

            bs = batch["image"].size(0)
            total_loss += tot_loss.item() * bs
            total_f_loss += f_loss.item() * bs
            total_r_loss += r_loss.item() * bs
            n_samples += bs

        return total_loss / n_samples, total_f_loss / n_samples, total_r_loss / n_samples

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        total_loss = 0.0
        n_samples = 0
        y_f_true, y_f_pred = [], []
        y_r_true, y_r_pred = [], []

        for batch in loader:
            with autocast(device_type="cuda" if self.device == "cuda" else "cpu", enabled=self.use_amp):
                tot_loss, _, _, f_logits, r_logits = self._forward_step(batch)

            bs = batch["image"].size(0)
            total_loss += tot_loss.item() * bs
            n_samples += bs

            if f_logits is not None:
                preds = f_logits.argmax(dim=1).cpu().numpy()
                y_f_pred.extend(preds)
                y_f_true.extend(batch["fruit_id"].numpy())

            if r_logits is not None:
                if self.model_type == "resnet50_softmax_ripeness":
                    preds = r_logits.argmax(dim=1).cpu().numpy()
                else:
                    preds = coral_predict(r_logits).cpu().numpy()
                y_r_pred.extend(preds)
                y_r_true.extend(batch["ripeness_id"].numpy())

        f_metrics = compute_fruit_metrics(y_f_true, y_f_pred) if y_f_true else {}
        r_metrics = compute_ripeness_metrics(y_r_true, y_r_pred) if y_r_true else {}

        return total_loss / n_samples, f_metrics, r_metrics

    def fit(self):
        print(f"=== Starting Training [{self.exp_name}] on {self.device.upper()} ===")
        frozen = self.freeze_epochs > 0
        optimizer = torch.optim.AdamW(
            get_parameter_groups(self.model, self.model_type, self.lr, frozen=frozen),
            weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=1e-6)

        best_score = float("inf")
        bad_epochs = 0
        best_epoch = 0
        train_start = time.time()

        log_file = os.path.join(self.log_dir, f"{self.exp_name}.csv")
        with open(log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "phase", "train_loss", "val_loss",
                "fruit_acc", "fruit_f1", "ripeness_acc", "ripeness_mae", "ripeness_qwk", "lr", "epoch_sec"
            ])

        for ep in range(self.epochs):
            t0 = time.time()
            if frozen and ep == self.freeze_epochs:
                frozen = False
                print(f"[Epoch {ep}] Unfreezing backbone for fine-tuning...")
                optimizer = torch.optim.AdamW(
                    get_parameter_groups(self.model, self.model_type, self.lr, frozen=False),
                    weight_decay=self.weight_decay
                )
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=self.epochs - ep, eta_min=1e-6
                )

            tr_loss, _, _ = self.train_epoch(optimizer)
            val_loss, val_f, val_r = self.evaluate(self.val_loader)
            scheduler.step()

            # Metric logging
            f_acc = val_f.get("accuracy", 0.0)
            f_f1 = val_f.get("f1_macro", 0.0)
            r_acc = val_r.get("accuracy", 0.0)
            r_mae = val_r.get("mae", 0.0)
            r_qwk = val_r.get("qwk", 0.0)

            cur_lr = optimizer.param_groups[0]["lr"]
            sec = time.time() - t0
            phase = "frozen" if frozen else "fine-tuning"

            with open(log_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    ep, phase, f"{tr_loss:.4f}", f"{val_loss:.4f}",
                    f"{f_acc:.4f}", f"{f_f1:.4f}", f"{r_acc:.4f}", f"{r_mae:.4f}", f"{r_qwk:.4f}",
                    f"{cur_lr:.2e}", f"{sec:.1f}"
                ])

            print(
                f"Epoch {ep:2d} [{phase:11s}] TrainLoss: {tr_loss:.3f} | ValLoss: {val_loss:.3f} | "
                f"FruitAcc: {f_acc*100:.1f}% | RipeAcc: {r_acc*100:.1f}% (MAE {r_mae:.3f}) | {sec:.1f}s"
            )

            # Combined score for model checkpointing: validation loss + MAE
            val_score = val_loss + (r_mae if r_mae > 0 else 0.0)
            if val_score < best_score - 1e-4:
                best_score = val_score
                best_epoch = ep
                bad_epochs = 0
                torch.save(self.model.state_dict(), os.path.join(self.ckpt_dir, "best.pt"))
            elif not frozen:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    print(f"Early stopping triggered at epoch {ep} (patience={self.patience})")
                    break

        train_time = time.time() - train_start
        max_mem_gb = torch.cuda.max_memory_allocated() / 1e9 if self.device == "cuda" else 0.0

        # Load best weights and evaluate on test set
        best_pt = os.path.join(self.ckpt_dir, "best.pt")
        if os.path.exists(best_pt):
            self.model.load_state_dict(torch.load(best_pt, map_location=self.device, weights_only=True))

        test_loss, test_f, test_r = self.evaluate(self.test_loader)
        print("=" * 60)
        print(f"TEST RESULTS FOR [{self.exp_name}]:")
        if test_f:
            print(f"Fruit ID   -> Accuracy: {test_f['accuracy']*100:.2f}% | Macro F1: {test_f['f1_macro']:.4f}")
        if test_r:
            print(f"Ripeness   -> Accuracy: {test_r['accuracy']*100:.2f}% | MAE: {test_r['mae']:.4f} | QWK: {test_r['qwk']:.4f} | Severe: {test_r['severe_error_rate']*100:.2f}%")
        print(f"Best Epoch: {best_epoch} | Train Time: {train_time:.1f}s | Max VRAM: {max_mem_gb:.2f} GB")
        print("=" * 60)

        return {
            "exp_name": self.exp_name,
            "model_type": self.model_type,
            "best_epoch": best_epoch,
            "train_time_sec": round(train_time, 1),
            "max_vram_gb": round(max_mem_gb, 3),
            "test_loss": round(test_loss, 4),
            "fruit_accuracy": round(test_f.get("accuracy", 0.0), 4),
            "fruit_precision_macro": round(test_f.get("precision_macro", 0.0), 4),
            "fruit_recall_macro": round(test_f.get("recall_macro", 0.0), 4),
            "fruit_f1_macro": round(test_f.get("f1_macro", 0.0), 4),
            "fruit_f1_weighted": round(test_f.get("f1_weighted", 0.0), 4),
            "ripeness_accuracy": round(test_r.get("accuracy", 0.0), 4),
            "ripeness_precision_macro": round(test_r.get("precision_macro", 0.0), 4),
            "ripeness_recall_macro": round(test_r.get("recall_macro", 0.0), 4),
            "ripeness_f1_macro": round(test_r.get("f1_macro", 0.0), 4),
            "ripeness_mae": round(test_r.get("mae", 0.0), 4),
            "ripeness_qwk": round(test_r.get("qwk", 0.0), 4),
            "ripeness_adjacent_error": round(test_r.get("adjacent_error_rate", 0.0), 4),
            "ripeness_severe_error": round(test_r.get("severe_error_rate", 0.0), 4),
            "checkpoint_path": best_pt
        }

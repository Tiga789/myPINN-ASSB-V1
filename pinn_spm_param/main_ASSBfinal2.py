from __future__ import annotations

import argparse
import csv
import math
import shutil
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

UTIL_DIR = Path(__file__).resolve().parent / "util"
if str(UTIL_DIR) not in sys.path:
    sys.path.insert(0, str(UTIL_DIR))

from _losses_ASSBfinal2 import compute_batch_loss_ASSBfinal2
from init_pinn_ASSBfinal2 import checkpoint_payload_ASSBfinal2, initialize_ASSBfinal2, save_config_and_meta_ASSBfinal2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ASSBfinal2 physics-only discrete-step PINN training")
    parser.add_argument("-i", "--input_file", default="input_physics_only_ASSBfinal2", help="Input config path")
    return parser.parse_args()


def _make_lr_scheduler(optimizer: torch.optim.Optimizer, lr_start: float, lr_final: float, epochs: int):
    if epochs <= 1:
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    gamma = math.exp(math.log(max(lr_final, 1e-12) / max(lr_start, 1e-12)) / max(epochs - 1, 1))
    return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)


def main() -> None:
    args = parse_args()
    cfg, ctx, model = initialize_ASSBfinal2(args.input_file)

    model_id = int(cfg["ID"])
    model_dir = Path(f"ModelFin_{model_id}")
    log_dir = Path(f"LogFin_{model_id}")
    if model_dir.exists():
        shutil.rmtree(model_dir)
    if log_dir.exists():
        shutil.rmtree(log_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    save_config_and_meta_ASSBfinal2(model_dir, cfg, ctx)

    step_idx_all = torch.arange(ctx["n_steps"], dtype=torch.long)
    batch_size = int(cfg["BATCH_SIZE_STEPS"])
    shuffle = bool(cfg["SHUFFLE_STEPS"])
    loader = DataLoader(TensorDataset(step_idx_all), batch_size=batch_size, shuffle=shuffle, drop_last=False)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["LEARNING_RATE"]),
        weight_decay=float(cfg.get("WEIGHT_DECAY", 0.0)),
    )
    scheduler = _make_lr_scheduler(optimizer, float(cfg["LEARNING_RATE"]), float(cfg["LEARNING_RATE_FINAL"]), int(cfg["EPOCHS"]))

    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_epoch = -1
    csv_path = log_dir / "history_ASSBfinal2.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=["epoch", "lr", "total", "step_a", "step_c", "bound_a", "bound_c", "seconds"])
        writer.writeheader()

        print("INFO: USING DISCRETE-STEP ASSB FINAL2 TRAINING")
        print(f"INFO: Device = {ctx['device']}")
        print("INFO: INT loss is ACTIVE")
        print("INFO: BOUND loss is ACTIVE")
        print("INFO: DATA loss is INACTIVE")
        print("INFO: REG loss is INACTIVE")
        print(f"Num trainable param =  {model.count_trainable_parameters()}")
        print(f"INFO: n_steps per epoch = {ctx['n_steps']} | batch_size_steps = {batch_size}")

        t0 = time.time()
        for epoch in range(1, int(cfg["EPOCHS"]) + 1):
            epoch_start = time.time()
            model.train()
            accum = {"total": 0.0, "step_a": 0.0, "step_c": 0.0, "bound_a": 0.0, "bound_c": 0.0}
            n_seen = 0
            for (step_batch,) in loader:
                step_batch = step_batch.to(ctx["device"])
                optimizer.zero_grad(set_to_none=True)
                breakdown = compute_batch_loss_ASSBfinal2(model, step_batch, ctx, cfg)
                breakdown.total.backward()
                clip_norm = float(cfg.get("GRAD_CLIP_NORM", 0.0))
                if clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
                optimizer.step()

                batch_n = int(step_batch.numel())
                vals = breakdown.as_float_dict()
                for k in accum:
                    accum[k] += vals[k] * batch_n
                n_seen += batch_n

            scheduler.step()
            epoch_seconds = time.time() - epoch_start
            lr_now = float(optimizer.param_groups[0]["lr"])
            row = {
                "epoch": epoch,
                "lr": lr_now,
                "seconds": epoch_seconds,
            }
            for k in accum:
                row[k] = accum[k] / max(n_seen, 1)
            history.append({k: float(v) for k, v in row.items()})
            writer.writerow(row)
            fcsv.flush()

            if row["total"] < best_loss:
                best_loss = float(row["total"])
                best_epoch = int(epoch)
                torch.save(
                    checkpoint_payload_ASSBfinal2(cfg, ctx, model, epoch, best_loss, history),
                    model_dir / "best.pt",
                )

            if epoch == 1 or epoch % int(cfg["SAVE_EVERY"]) == 0 or epoch == int(cfg["EPOCHS"]):
                torch.save(
                    checkpoint_payload_ASSBfinal2(cfg, ctx, model, epoch, best_loss, history),
                    model_dir / "last.pt",
                )

            if epoch == 1 or epoch % int(cfg["LOG_EVERY"]) == 0 or epoch == int(cfg["EPOCHS"]):
                print(
                    f"Epoch {epoch:4d}/{int(cfg['EPOCHS'])} | lr={lr_now:.3e} | "
                    f"total={row['total']:.6e} step_a={row['step_a']:.6e} step_c={row['step_c']:.6e} "
                    f"bound_a={row['bound_a']:.6e} bound_c={row['bound_c']:.6e} | {epoch_seconds:.2f}s"
                )

        total_time = time.time() - t0

    print(f"Best epoch {best_epoch}")
    print(f"Total time {total_time:.2f}s")
    print(f"Unweighted loss {best_loss}")


if __name__ == "__main__":
    main()

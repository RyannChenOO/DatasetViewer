"""Train, evaluate, plot, and export the navigation MLP."""

from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from generate_data import generate
from model import ACTION_NAMES, IMAGE_SIZE, LinearBaseline, NavigationMLP


ROOT = Path(__file__).resolve().parents[1]
SEED = 7400


def set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def preprocess_image(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32).reshape(-1) / 255.0


def load_dataset(csv_path: Path) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    cache: dict[str, np.ndarray] = {}
    grouped: dict[str, list[tuple[np.ndarray, int]]] = {"train": [], "val": [], "test": []}
    with csv_path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            image_path = row["rgb"]
            if image_path not in cache:
                cache[image_path] = preprocess_image(ROOT / image_path)
            goal = np.asarray(
                [float(row["goal_distance_m"]) / 10.0, float(row["goal_angle_rad"]) / math.pi],
                dtype=np.float32,
            )
            features = np.concatenate([cache[image_path], goal])
            grouped[row["split"]].append((features, int(row["action_id"])))
    result = {}
    for split, samples in grouped.items():
        x = torch.tensor(np.stack([sample[0] for sample in samples]), dtype=torch.float32)
        y = torch.tensor([sample[1] for sample in samples], dtype=torch.long)
        result[split] = (x, y)
    return result


@torch.no_grad()
def evaluate(model: nn.Module, data: tuple[torch.Tensor, torch.Tensor], loss_fn: nn.Module) -> tuple[float, float]:
    model.eval()
    x, y = data
    logits = model(x)
    return float(loss_fn(logits, y)), float((logits.argmax(dim=1) == y).float().mean())


def train_model(model: nn.Module, data: dict[str, tuple[torch.Tensor, torch.Tensor]], epochs: int, lr: float) -> dict[str, list[float]]:
    train_x, train_y = data["train"]
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=128, shuffle=True)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"train_loss": [], "val_loss": [], "train_accuracy": [], "val_accuracy": []}
    for _ in range(epochs):
        model.train()
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
        train_loss, train_accuracy = evaluate(model, data["train"], loss_fn)
        val_loss, val_accuracy = evaluate(model, data["val"], loss_fn)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_accuracy"].append(val_accuracy)
    return history


def export_browser_weights(model: NavigationMLP, metrics: dict[str, object], output: Path) -> None:
    layers = [layer for layer in model.network if isinstance(layer, nn.Linear)]
    payload = {
        "architecture": [194, 64, 32, 4],
        "image_size": IMAGE_SIZE,
        "actions": ACTION_NAMES,
        "normalization": {"pixels": "value / 255", "goal_distance": "meters / 10", "goal_angle": "radians / pi"},
        "layers": [
            {"weight": layer.weight.detach().tolist(), "bias": layer.bias.detach().tolist()}
            for layer in layers
        ],
        "metrics": metrics,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")


def save_plot(history: dict[str, list[float]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, (loss_axis, accuracy_axis) = plt.subplots(1, 2, figsize=(10, 3.6), dpi=160)
    loss_axis.plot(epochs, history["train_loss"], label="Train", color="#173f35", linewidth=2)
    loss_axis.plot(epochs, history["val_loss"], label="Validation", color="#d37c52", linewidth=2)
    loss_axis.set(title="Cross-entropy loss", xlabel="Epoch", ylabel="Loss")
    loss_axis.legend(frameon=False)
    accuracy_axis.plot(epochs, history["train_accuracy"], label="Train", color="#173f35", linewidth=2)
    accuracy_axis.plot(epochs, history["val_accuracy"], label="Validation", color="#d37c52", linewidth=2)
    accuracy_axis.set(title="Action accuracy", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1.02))
    accuracy_axis.legend(frameon=False)
    figure.suptitle("Navigation MLP training monitor", fontsize=13, fontweight="bold")
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    set_seed()
    csv_path = ROOT / "data" / "navigation_samples.csv"
    if not csv_path.exists():
        csv_path = generate(ROOT, episodes=120, steps=24, seed=SEED)
    data = load_dataset(csv_path)
    loss_fn = nn.CrossEntropyLoss()

    mlp = NavigationMLP()
    mlp_history = train_model(mlp, data, epochs=90, lr=0.0025)
    mlp_test_loss, mlp_test_accuracy = evaluate(mlp, data["test"], loss_fn)

    set_seed()
    baseline = LinearBaseline()
    baseline_history = train_model(baseline, data, epochs=90, lr=0.0025)
    baseline_test_loss, baseline_test_accuracy = evaluate(baseline, data["test"], loss_fn)

    counts = {split: int(values[1].shape[0]) for split, values in data.items()}
    metrics = {
        "dataset_samples": sum(counts.values()),
        "split_samples": counts,
        "epochs": 90,
        "learning_rate": 0.0025,
        "optimizer": "Adam",
        "loss": "CrossEntropyLoss",
        "train_accuracy": round(mlp_history["train_accuracy"][-1], 4),
        "validation_accuracy": round(mlp_history["val_accuracy"][-1], 4),
        "test_accuracy": round(mlp_test_accuracy, 4),
        "test_loss": round(mlp_test_loss, 4),
        "linear_baseline_test_accuracy": round(baseline_test_accuracy, 4),
        "linear_baseline_test_loss": round(baseline_test_loss, 4),
    }

    model_dir = ROOT / "model"
    model_dir.mkdir(exist_ok=True)
    torch.save({"state_dict": mlp.state_dict(), "metrics": metrics}, model_dir / "navigation_mlp.pt")
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (model_dir / "training_history.json").write_text(json.dumps(mlp_history), encoding="utf-8")
    export_browser_weights(mlp, metrics, ROOT / "public" / "model" / "navigation_mlp.json")
    save_plot(mlp_history, ROOT / "public" / "model" / "training_curves.png")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

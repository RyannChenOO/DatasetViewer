"""Run the controlled HW3 baseline, proposed model, and no-gate ablation."""

from __future__ import annotations

import json
import random
import time
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from hw3_models import GoalGatedResidualMLP, HW2Baseline, UngatedResidualMLP
from model import ACTION_NAMES
from train import load_dataset


ROOT = Path(__file__).resolve().parents[1]
SEEDS = [7400, 7401, 7402, 7403, 7404]
EPOCHS = 90
LEARNING_RATE = 0.0025
BATCH_SIZE = 128

MODEL_SPECS: dict[str, tuple[str, Callable[[], nn.Module]]] = {
    "baseline": ("HW2 baseline", HW2Baseline),
    "gated_residual": ("Goal-gated residual", GoalGatedResidualMLP),
    "ungated_residual": ("No-gate ablation", UngatedResidualMLP),
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def evaluate(model: nn.Module, split: tuple[torch.Tensor, torch.Tensor]) -> dict[str, object]:
    model.eval()
    x, y = split
    logits = model(x)
    predictions = logits.argmax(dim=1)
    loss = nn.functional.cross_entropy(logits, y).item()
    confusion = torch.zeros((len(ACTION_NAMES), len(ACTION_NAMES)), dtype=torch.int64)
    for truth, prediction in zip(y, predictions):
        confusion[int(truth), int(prediction)] += 1

    per_class_accuracy = []
    f1_scores = []
    for index in range(len(ACTION_NAMES)):
        true_positive = int(confusion[index, index])
        support = int(confusion[index].sum())
        predicted = int(confusion[:, index].sum())
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        f1_scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
        per_class_accuracy.append(true_positive / support if support else 0.0)

    return {
        "loss": loss,
        "accuracy": float((predictions == y).float().mean()),
        "macro_f1": float(np.mean(f1_scores)),
        "per_class_accuracy": dict(zip(ACTION_NAMES, per_class_accuracy)),
        "confusion_matrix": confusion.tolist(),
    }


def train_one(model: nn.Module, data: dict[str, tuple[torch.Tensor, torch.Tensor]]) -> dict[str, list[float]]:
    train_x, train_y = data["train"]
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    history = {"train_loss": [], "val_loss": [], "train_accuracy": [], "val_accuracy": []}

    for _ in range(EPOCHS):
        model.train()
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
        train_metrics = evaluate(model, data["train"])
        val_metrics = evaluate(model, data["val"])
        history["train_loss"].append(float(train_metrics["loss"]))
        history["val_loss"].append(float(val_metrics["loss"]))
        history["train_accuracy"].append(float(train_metrics["accuracy"]))
        history["val_accuracy"].append(float(val_metrics["accuracy"]))
    return history


@torch.no_grad()
def measure_latency_ms(model: nn.Module, sample: torch.Tensor, repeats: int = 1500) -> float:
    model.eval()
    for _ in range(100):
        model(sample)
    start = time.perf_counter()
    for _ in range(repeats):
        model(sample)
    return (time.perf_counter() - start) * 1000 / repeats


@torch.no_grad()
def gate_statistics(model: GoalGatedResidualMLP, inputs: torch.Tensor) -> dict[str, float]:
    _, gates = model(inputs, return_gate=True)
    return {
        "mean": float(gates.mean()),
        "std": float(gates.std()),
        "minimum": float(gates.min()),
        "maximum": float(gates.max()),
        "saturation_fraction": float(((gates < 0.1) | (gates > 0.9)).float().mean()),
    }


def aggregate(values: list[float]) -> dict[str, float]:
    return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1))}


def save_training_plot(runs: dict[str, list[dict[str, object]]], output: Path) -> None:
    colors = {"baseline": "#58766a", "gated_residual": "#e27645", "ungated_residual": "#788bd4"}
    epochs = np.arange(1, EPOCHS + 1)
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, (loss_axis, accuracy_axis) = plt.subplots(1, 2, figsize=(10.8, 4), dpi=180)
    for key, (label, _) in MODEL_SPECS.items():
        histories = [run["history"] for run in runs[key]]
        for axis, metric in ((loss_axis, "val_loss"), (accuracy_axis, "val_accuracy")):
            matrix = np.asarray([history[metric] for history in histories], dtype=float)
            mean, std = matrix.mean(axis=0), matrix.std(axis=0, ddof=1)
            axis.plot(epochs, mean, label=label, color=colors[key], linewidth=2)
            axis.fill_between(epochs, mean - std, mean + std, color=colors[key], alpha=0.13)
    loss_axis.set(title="Validation cross-entropy", xlabel="Epoch", ylabel="Loss")
    accuracy_axis.set(title="Validation accuracy", xlabel="Epoch", ylabel="Accuracy", ylim=(0.6, 1.01))
    loss_axis.legend(frameon=False, fontsize=8)
    accuracy_axis.legend(frameon=False, fontsize=8)
    figure.suptitle("HW3 controlled comparison (mean ± 1 SD, 5 seeds)", fontweight="bold")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def save_confusion_plot(runs: dict[str, list[dict[str, object]]], output: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(1, 3, figsize=(11.4, 3.5), dpi=180)
    for axis, (key, (label, _)) in zip(axes, MODEL_SPECS.items()):
        matrices = np.asarray([run["test"]["confusion_matrix"] for run in runs[key]], dtype=float)
        matrix = matrices.mean(axis=0)
        normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
        image = axis.imshow(normalized, vmin=0, vmax=1, cmap="YlGn")
        axis.set_title(label, fontsize=10, fontweight="bold")
        axis.set_xticks(range(4), ["FWD", "LEFT", "RIGHT", "STOP"], rotation=35, ha="right", fontsize=7)
        axis.set_yticks(range(4), ["FWD", "LEFT", "RIGHT", "STOP"], fontsize=7)
        axis.set_xlabel("Predicted", fontsize=8)
        if axis is axes[0]:
            axis.set_ylabel("True label", fontsize=8)
        for row in range(4):
            for column in range(4):
                axis.text(column, row, f"{normalized[row, column] * 100:.0f}%", ha="center", va="center", fontsize=7,
                          color="white" if normalized[row, column] > 0.58 else "#18332a")
    figure.colorbar(image, ax=axes, fraction=0.022, pad=0.02, label="Row-normalized fraction")
    figure.suptitle("Mean test confusion matrices across 5 seeds", fontweight="bold")
    figure.subplots_adjust(left=0.06, right=0.92, bottom=0.18, top=0.80, wspace=0.28)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def save_accuracy_plot(summary: dict[str, object], output: Path) -> None:
    keys = list(MODEL_SPECS)
    labels = [MODEL_SPECS[key][0] for key in keys]
    means = [summary["models"][key]["test_accuracy"]["mean"] * 100 for key in keys]
    stds = [summary["models"][key]["test_accuracy"]["std"] * 100 for key in keys]
    colors = ["#58766a", "#e27645", "#788bd4"]
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axis = plt.subplots(figsize=(7.4, 3.6), dpi=180)
    bars = axis.bar(labels, means, yerr=stds, capsize=5, color=colors, width=0.62)
    axis.set(ylabel="Test accuracy (%)", ylim=(75, 100), title="HW3 model comparison (5 seeds)")
    axis.grid(axis="x", visible=False)
    for bar, mean, std in zip(bars, means, stds):
        axis.text(bar.get_x() + bar.get_width() / 2, mean + std + 0.5, f"{mean:.2f} ± {std:.2f}", ha="center", fontsize=8, fontweight="bold")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    torch.set_num_threads(1)
    data = load_dataset(ROOT / "data" / "navigation_samples.csv")
    all_runs: dict[str, list[dict[str, object]]] = {key: [] for key in MODEL_SPECS}
    best_checkpoints: dict[str, tuple[float, int, dict[str, torch.Tensor]]] = {}

    for model_key, (label, factory) in MODEL_SPECS.items():
        for seed in SEEDS:
            set_seed(seed)
            model = factory()
            history = train_one(model, data)
            train_metrics = evaluate(model, data["train"])
            val_metrics = evaluate(model, data["val"])
            test_metrics = evaluate(model, data["test"])
            run: dict[str, object] = {
                "seed": seed,
                "history": history,
                "train": train_metrics,
                "validation": val_metrics,
                "test": test_metrics,
                "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                "latency_ms": measure_latency_ms(model, data["test"][0][:1]),
            }
            if isinstance(model, GoalGatedResidualMLP):
                run["gate_statistics"] = gate_statistics(model, data["test"][0])
            all_runs[model_key].append(run)

            score = float(val_metrics["accuracy"])
            if model_key not in best_checkpoints or score > best_checkpoints[model_key][0]:
                best_checkpoints[model_key] = (
                    score,
                    seed,
                    {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()},
                )
            print(f"{label:22s} seed={seed} test={float(test_metrics['accuracy']) * 100:.2f}%")

    counts = {split: int(values[1].shape[0]) for split, values in data.items()}
    summary: dict[str, object] = {
        "experiment": "Goal-Conditioned Gated Residual Fusion",
        "hypothesis": "Separating RGB and goal encoding, then using the goal to gate a residual visual correction, will preserve visual context while making the two goal features harder to ignore.",
        "controls": {
            "seeds": SEEDS,
            "epochs": EPOCHS,
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "loss": "CrossEntropyLoss",
            "split": counts,
        },
        "models": {},
    }

    for key, (label, _) in MODEL_SPECS.items():
        runs = all_runs[key]
        model_summary: dict[str, object] = {
            "label": label,
            "parameter_count": runs[0]["parameter_count"],
            "test_accuracy": aggregate([float(run["test"]["accuracy"]) for run in runs]),
            "test_macro_f1": aggregate([float(run["test"]["macro_f1"]) for run in runs]),
            "test_loss": aggregate([float(run["test"]["loss"]) for run in runs]),
            "validation_accuracy": aggregate([float(run["validation"]["accuracy"]) for run in runs]),
            "latency_ms": aggregate([float(run["latency_ms"]) for run in runs]),
            "per_class_accuracy": {
                action: aggregate([float(run["test"]["per_class_accuracy"][action]) for run in runs])
                for action in ACTION_NAMES
            },
            "best_checkpoint_seed": best_checkpoints[key][1],
        }
        if key == "gated_residual":
            model_summary["gate_statistics"] = {
                field: aggregate([float(run["gate_statistics"][field]) for run in runs])
                for field in ("mean", "std", "minimum", "maximum", "saturation_fraction")
            }
        summary["models"][key] = model_summary

    model_dir = ROOT / "model"
    public_dir = ROOT / "public" / "model"
    model_dir.mkdir(exist_ok=True)
    public_dir.mkdir(parents=True, exist_ok=True)
    for key, (_, factory) in MODEL_SPECS.items():
        score, seed, state_dict = best_checkpoints[key]
        torch.save({"state_dict": state_dict, "seed": seed, "validation_accuracy": score}, model_dir / f"hw3_{key}.pt")

    (model_dir / "hw3_results.json").write_text(json.dumps({"summary": summary, "runs": all_runs}, indent=2), encoding="utf-8")
    (public_dir / "hw3_summary.json").write_text(json.dumps(summary, separators=(",", ":")), encoding="utf-8")
    save_training_plot(all_runs, public_dir / "hw3_training_curves.png")
    save_confusion_plot(all_runs, public_dir / "hw3_confusion_matrices.png")
    save_accuracy_plot(summary, public_dir / "hw3_accuracy_comparison.png")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

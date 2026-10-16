"""Create an emulated PointNav dataset around public Habitat RGB observations.

The RGB frames come from Habitat-Lab's public PointNav visualization. Goal
distance, relative goal angle, and expert actions are generated for this class
assignment; they are not claimed to be metadata recovered from the source GIF.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image

from model import ACTION_NAMES, IMAGE_SIZE


def image_features(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32).reshape(-1) / 255.0


def visual_affordances(features: np.ndarray) -> tuple[float, float, float]:
    image = features.reshape(IMAGE_SIZE, IMAGE_SIZE, 3).mean(axis=2)
    lower = image[IMAGE_SIZE // 2 :, :]
    left_open = float(lower[:, : IMAGE_SIZE // 3].mean())
    center_open = float(lower[:, IMAGE_SIZE // 3 : 2 * IMAGE_SIZE // 3].mean())
    right_open = float(lower[:, 2 * IMAGE_SIZE // 3 :].mean())
    return left_open, center_open, right_open


def expert_action(distance: float, angle: float, affordances: tuple[float, float, float]) -> str:
    """A deterministic, interpretable emulated expert policy.

    The policy stops near an aligned goal, turns for large angular error, and
    uses image brightness as a simple local free-space cue when the goal is
    roughly ahead. The last branch creates a visual/goal interaction that a
    hidden-layer MLP can represent better than a single linear layer.
    """

    left_open, center_open, right_open = affordances
    if distance < 0.72 and abs(angle) < 0.32:
        return "STOP"
    if angle < -0.34:
        return "TURN_LEFT"
    if angle > 0.34:
        return "TURN_RIGHT"
    adaptive_threshold = 0.36 + 0.11 * (distance / 10.0)
    if center_open > adaptive_threshold:
        return "MOVE_FORWARD"
    return "TURN_LEFT" if left_open >= right_open else "TURN_RIGHT"


def generate(root: Path, episodes: int, steps: int, seed: int) -> Path:
    rng = random.Random(seed)
    frame_dir = root / "public" / "habitat" / "frames"
    output = root / "data" / "navigation_samples.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame_paths = sorted(frame_dir.glob("frame-*.jpg"))
    if not frame_paths:
        raise FileNotFoundError(f"No Habitat frames found in {frame_dir}")

    cache = {path: image_features(path) for path in frame_paths}
    rows: list[dict[str, str | int | float]] = []
    for episode in range(episodes):
        split = "train" if episode < int(episodes * 0.70) else "val" if episode < int(episodes * 0.85) else "test"
        phase = rng.uniform(-math.pi, math.pi)
        start_distance = rng.uniform(5.5, 10.0)
        for step in range(steps):
            path = frame_paths[(episode * 7 + step * 3) % len(frame_paths)]
            progress = step / max(1, steps - 1)
            # Include near-goal samples in every episode so STOP is represented.
            distance = max(0.12, start_distance * (1.0 - progress) + rng.uniform(-0.18, 0.18))
            if step >= steps - 3:
                distance = rng.uniform(0.12, 0.68)
                angle = rng.uniform(-0.28, 0.28)
            else:
                angle = 1.35 * math.sin(phase + step * 0.47) + rng.uniform(-0.38, 0.38)
                angle = max(-math.pi, min(math.pi, angle))
            action = expert_action(distance, angle, visual_affordances(cache[path]))
            rows.append(
                {
                    "episode_id": f"emulated_{episode:03d}",
                    "step": step,
                    "split": split,
                    "rgb": str(path.relative_to(root)),
                    "goal_distance_m": round(distance, 5),
                    "goal_angle_rad": round(angle, 6),
                    "action": action,
                    "action_id": ACTION_NAMES.index(action),
                }
            )

    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} samples from {episodes} emulated episodes to {output}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=120)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=7400)
    args = parser.parse_args()
    generate(Path(__file__).resolve().parents[1], args.episodes, args.steps, args.seed)

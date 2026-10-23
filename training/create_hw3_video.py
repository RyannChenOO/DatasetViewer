"""Create a narrated, under-three-minute HW3 demo from the local web application."""

from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

from playwright.sync_api import Page, sync_playwright


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT.parent / "tmp" / "hw3-video"
OUTPUT = ROOT.parent / "output" / "video" / "Habitat_Navigation_Learner_HW3_Demo.mp4"
FPS = 3
SCENES = [
    {
        "caption": "HW3 · Navigation design experiment",
        "selector": ".workspace-grid",
        "text": "Hello, this is my Homework Three design experiment for Habitat next-action prediction. My task uses the current RGB image, goal distance, and relative goal angle to predict move forward, turn left, turn right, or stop.",
    },
    {
        "caption": "Limitation · Early fusion can dilute the goal signal",
        "selector": ".model-monitor",
        "text": "My Homework Two baseline immediately concatenated 192 image values with only two goal values. I was concerned that the high-dimensional image could dominate the compact but important goal signal. The baseline also had no direct path that preserved visual information while goal context modified it.",
    },
    {
        "caption": "Proposal · Goal-conditioned gated residual fusion",
        "selector": ".architecture-flow",
        "text": "My proposed design is goal-conditioned gated residual fusion. The image and goal are first encoded by separate branches. A joint branch predicts a bounded correction. The goal branch also predicts a sigmoid gate, and the fused feature is the visual representation plus the gate times the correction. I expected this to preserve visual context while making goal information harder to ignore. A possible failure was gate saturation.",
    },
    {
        "caption": "Controlled experiment · 3 models × 5 seeds",
        "selector": ".experiment-grid",
        "text": "I ran a controlled comparison among the original Homework Two baseline, the full gated model, and a no-gate residual ablation. Every model used the same 2,880 samples, episode split, preprocessing, 90 epochs, Adam learning rate, batch size, and cross-entropy loss. I repeated each experiment with five random seeds.",
    },
    {
        "caption": "Result · 95.60% → 98.70% test accuracy",
        "selector": ".experiment-grid",
        "text": "The baseline reached 95.60 percent mean test accuracy. The full design reached 98.70 percent, an improvement of 3.10 percentage points, and its seed-to-seed variation was lower. The largest class-level improvement was move forward, from 76.3 to 93.0 percent.",
    },
    {
        "caption": "Evidence · Faster convergence across five seeds",
        "selector": ".evidence-grid figure:first-child",
        "text": "The validation curves show that both residual variants learned faster and converged above the baseline. The mean confusion matrices also show that most remaining mistakes are forward actions being classified as turns.",
    },
    {
        "caption": "Ablation · Most gain remains without the gate",
        "selector": ".evidence-grid figure:nth-child(2)",
        "text": "The ablation changed my interpretation. Without the gate, the model still reached 98.38 percent, only 0.32 points below the full design. So separate modality encoders and the residual path explain most of the gain, not gating alone.",
    },
    {
        "caption": "Failure analysis · 52.1% of gates near saturation",
        "selector": ".finding-strip",
        "text": "The gate averaged 0.853, and 52.1 percent of activations were near saturation. It often became an almost-open switch, which matches my predicted failure mode. The full model also has 42 percent more parameters and about 2.8 times the measured single-sample latency, so extra capacity is a confounding factor.",
    },
    {
        "caption": "Lesson · Separate evidence from explanation",
        "selector": ".finding-strip",
        "text": "I learned that explicit modality handling and residual fusion are useful for this task, but adding a learned mechanism does not prove that mechanism caused the improvement. My next experiment would use a parameter-matched control and gate-entropy regularization, then evaluate on true scene-disjoint Habitat rollouts. The code, per-seed results, visualizations, and report are all included in the repository. Thank you.",
    },
]


def media_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        check=True, capture_output=True, text=True,
    )
    return float(result.stdout.strip())


def prepare_audio() -> list[float]:
    audio_dir = WORK / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    concat = []
    durations = []
    for index, scene in enumerate(SCENES):
        aiff = audio_dir / f"scene-{index:02d}.aiff"
        wav = audio_dir / f"scene-{index:02d}.wav"
        subprocess.run(["say", "-v", "Samantha", "-r", "170", "-o", str(aiff), scene["text"]], check=True)
        subprocess.run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(aiff),
            "-af", "apad=pad_dur=0.75", "-ar", "44100", "-ac", "1", "-c:a", "pcm_s16le", str(wav),
        ], check=True)
        durations.append(media_duration(wav))
        concat.append(f"file '{wav.name}'")
    concat_path = audio_dir / "concat.txt"
    concat_path.write_text("\n".join(concat) + "\n", encoding="utf-8")
    subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0",
        "-i", str(concat_path), "-c:a", "pcm_s16le", str(WORK / "narration.wav"),
    ], check=True)
    (WORK / "timeline.json").write_text(json.dumps({"durations": durations, "total": sum(durations)}, indent=2), encoding="utf-8")
    return durations


def add_overlays(page: Page) -> None:
    page.evaluate(
        """() => {
          const style=document.createElement('style');
          style.textContent=`
            html{scroll-behavior:smooth}
            .demo-focus{outline:3px solid #df7b49!important;outline-offset:7px;box-shadow:0 0 0 10px rgba(223,123,73,.13)!important}
            #demo-caption{position:fixed;left:30px;bottom:25px;z-index:99999;max-width:760px;padding:12px 18px;border:1px solid rgba(215,249,155,.6);border-radius:10px;color:#f8fbf8;background:rgba(17,39,31,.94);box-shadow:0 10px 32px rgba(0,0,0,.28);font:700 16px Arial,sans-serif;letter-spacing:.015em;pointer-events:none}
            #demo-badge{position:fixed;right:30px;top:24px;z-index:99999;padding:8px 12px;color:#17372b;border-radius:8px;background:#d7f99b;font:800 11px Arial,sans-serif;letter-spacing:.12em;pointer-events:none}
          `;
          document.head.appendChild(style);
          const caption=document.createElement('div');caption.id='demo-caption';document.body.appendChild(caption);
          const badge=document.createElement('div');badge.id='demo-badge';badge.textContent='DS7400 · HW3';document.body.appendChild(badge);
        }"""
    )


def focus(page: Page, selector: str, caption: str) -> None:
    page.evaluate(
        """([selector, caption]) => {
          document.querySelectorAll('.demo-focus').forEach(el=>el.classList.remove('demo-focus'));
          const target=document.querySelector(selector);
          if(target){target.classList.add('demo-focus');target.scrollIntoView({behavior:'instant',block:'center'});}
          const label=document.querySelector('#demo-caption');if(label)label.textContent=caption;
        }""",
        [selector, caption],
    )
    page.wait_for_timeout(650)


def capture_scene(page: Page, frames_dir: Path, start_index: int, duration: float) -> int:
    frame_count = max(1, round(duration * FPS))
    start = time.monotonic()
    for offset in range(frame_count):
        target = start + offset / FPS
        remaining = target - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        page.screenshot(path=str(frames_dir / f"frame-{start_index + offset:05d}.jpg"), type="jpeg", quality=88)
    return start_index + frame_count


def record_frames(durations: list[float]) -> Path:
    frames_dir = WORK / f"frames-{datetime.now().strftime('%H%M%S')}"
    frames_dir.mkdir(parents=True, exist_ok=True)
    index = 0
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 720}, device_scale_factor=1)
        page.goto("http://127.0.0.1:3000/", wait_until="networkidle")
        add_overlays(page)
        for scene, duration in zip(SCENES, durations):
            focus(page, scene["selector"], scene["caption"])
            index = capture_scene(page, frames_dir, index, duration)
        browser.close()
    return frames_dir


def assemble(frames_dir: Path) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-framerate", str(FPS),
        "-i", str(frames_dir / "frame-%05d.jpg"), "-i", str(WORK / "narration.wav"),
        "-vf", "fps=30,format=yuv420p", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", "-shortest", str(OUTPUT),
    ], check=True)


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    durations = prepare_audio()
    frames_dir = record_frames(durations)
    assemble(frames_dir)
    print(json.dumps({"output": str(OUTPUT), "duration": media_duration(OUTPUT), "audio_duration": sum(durations)}, indent=2))


if __name__ == "__main__":
    main()

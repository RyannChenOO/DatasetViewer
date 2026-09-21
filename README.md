# Habitat Trajectory Annotator

An interactive prototype for displaying and annotating long embodied-navigation trajectories. The interface is designed for spatial-intelligence research where each trajectory contains synchronized egocentric observations and environment context.

**Live demo:** [locus-campus-spatial-viewer.ryannchenoo.chatgpt.site](https://locus-campus-spatial-viewer.ryannchenoo.chatgpt.site)

## Public sample data

The bundled images are sampled from the official Habitat-Lab PointNav top-down-map visualization:

- Project: [Habitat-Lab](https://github.com/facebookresearch/habitat-lab)
- Original asset: [`skokloster-castle.glb_3662.gif`](https://github.com/facebookresearch/habitat-lab/blob/main/docs/images/habitat-lab-tdmap-viz-images/skokloster-castle.glb_3662.gif)
- Related platform: [Habitat 3.0](https://aihabitat.org/habitat3/)

The original visualization provides synchronized RGB, depth, and top-down-map views. This prototype samples 30 observations at 3 fps. The action, collision, timing, and goal-distance fields shown in the interface are illustrative metadata for demonstrating the annotation workflow; they are not claimed as ground-truth fields recovered from the GIF.

No images or statistics from the *Lost on Campus* paper are used by this version.

## Why the interface is trajectory-first

Long horizontal data is shown at two scales:

1. **Overview track:** compresses the entire trajectory into a fixed-width strip and displays interval labels plus the current playhead.
2. **Detail filmstrip:** provides a horizontally scrollable sequence of readable RGB thumbnails. Selecting a frame synchronizes the large RGB observation, depth image, map, and metadata.

Separate action and event tracks make temporal patterns visible without squeezing more information into each thumbnail.

## Annotation model

### Frame-level annotations

- Point or bounding-box geometry
- Labels: landmark, obstacle, navigable area, localization cue, goal evidence, ambiguous region
- Coordinates are stored as percentages, so annotations remain aligned when the image is resized

### Trajectory-level annotations

- Whole-trajectory or frame-interval labels
- Labels: exploration, correct progress, wrong turn, collision recovery, goal recognition
- Every interval stores an inclusive start and end frame

Annotations live in the current browser session and can be exported to or imported from JSON. Export is the explicit persistence mechanism for this homework prototype; no account or backend database is required.

## Suggested real-data schema

```json
{
  "trajectory_id": "scene_episode_id",
  "scene_id": "scene.glb",
  "task": "pointnav",
  "steps": [
    {
      "step": 0,
      "rgb": "frames/000000.jpg",
      "depth": "depth/000000.png",
      "position": [0, 0, 0],
      "rotation": [0, 0, 0, 1],
      "action": "MOVE_FORWARD",
      "collision": false,
      "distance_to_goal": 8.8
    }
  ],
  "frame_annotations": [],
  "segment_annotations": []
}
```

For a real dataset, record this structure while rolling out an agent in Habitat-Sim or Habitat-Lab. Episode definitions alone usually contain scene, start state, goal, and optional shortest-path information; the RGB trajectory must be rendered and saved during rollout.

## Implemented interactions

- Frame selection from overview, filmstrip, action track, or event track
- Previous/next navigation and timed playback
- Synchronized RGB, depth, and top-down map
- Adjustable filmstrip thumbnail width for long trajectories
- Point and box annotations on individual RGB frames
- Interval annotations on the full trajectory
- Annotation deletion, JSON import, and JSON export
- Keyboard navigation with left/right arrow keys
- Responsive desktop and mobile layouts

## Run locally

Requires Node.js 22.13 or newer.

```bash
npm install
npm run dev
```

Create a production build with `npm run build`.

## Stack

- React 19 and TypeScript
- Vinext / Vite
- Tailwind CSS
- shadcn UI primitives
- Lucide icons

# LOCUS — Campus Spatial Episode Viewer

LOCUS is an interactive web-based viewer for embodied spatial intelligence research data. It is inspired by the *Lost on Campus* project, which evaluates Vision-Language Models in large-scale, photorealistic outdoor campus environments reconstructed with 3D Gaussian Splatting.

## Data being visualized

The included sample dataset contains egocentric campus observations, goal views, and bird's-eye-view route maps derived from the *Lost on Campus* research paper. The full research dataset contains more than 50,000 calibrated video frames across 15 outdoor scenes; this homework repository intentionally includes only a small representative sample.

You can also load a local image with the **Add image** button. Uploaded images stay in the browser and are not sent to a server.

## Implemented features

- Browse and switch between egocentric observations, goal images, and failure cases
- Inspect the matching BEV route map and navigation metadata
- Zoom with the slider, buttons, or mouse wheel
- Pan the selected image by dragging
- Move to previous/next views
- Load an additional local research image
- Add point annotations for landmarks, paths, obstacles, and goals
- Persist annotations in browser local storage
- Delete annotations and export all labels as JSON
- Responsive layout for desktop and smaller screens

## Run locally

Requirements: Node.js 22.13 or newer and npm.

```bash
npm install
npm run dev
```

Open the local URL printed in the terminal, usually `http://localhost:3000`.

Create a production build with:

```bash
npm run build
```

## Major libraries and frameworks

- React 19
- Vinext / Vite
- TypeScript
- Tailwind CSS
- shadcn UI components
- Lucide icons

## Annotation and backend status

- **Interactive annotation:** Implemented. Users can place category-coded point labels on every image, remove them, and export them as JSON.
- **Backend/database:** Not implemented. Annotation data is stored locally in the browser with `localStorage`, keeping the HW1 scope focused and the app easy to run.

## Research direction

This viewer is designed as the interface foundation for later deep-learning assignments. Future extensions could connect a VLM navigation agent, display model-predicted actions and confidence, compare trajectories, evaluate the six diagnostic spatial reasoning capabilities, or store benchmark episodes and predictions in a backend database.

## Data note

The bundled sample images are provided only to demonstrate the viewer using the author's supplied research paper. Replace or extend them with approved research data as appropriate before public distribution.

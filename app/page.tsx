/* oxlint-disable next/no-img-element */
'use client';

import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent,
} from 'react';
import {
  BrainCircuit,
  BoxSelect,
  ChevronLeft,
  ChevronRight,
  CircleDot,
  Download,
  Flag,
  Info,
  Map,
  MousePointer2,
  Pause,
  Play,
  Route,
  ScanLine,
  Sparkles,
  Trash2,
  Upload,
} from 'lucide-react';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';

const TOTAL_STEPS = 30;
const SAMPLE_RATE = 3;

const framePath = (step: number) =>
  `/habitat/frames/frame-${String(step + 1).padStart(3, '0')}.jpg`;
const depthPath = (step: number) =>
  `/habitat/depth/depth-${String(step + 1).padStart(3, '0')}.jpg`;
const mapPath = (step: number) =>
  `/habitat/maps/map-${String(step + 1).padStart(3, '0')}.jpg`;

type FrameLabel =
  | 'landmark'
  | 'obstacle'
  | 'navigable'
  | 'localization-cue'
  | 'goal-evidence'
  | 'ambiguous';
type DrawTool = 'inspect' | 'point' | 'box';
type SegmentLabel =
  | 'exploration'
  | 'correct-progress'
  | 'wrong-turn'
  | 'collision-recovery'
  | 'goal-recognition';
type ActionName = 'MOVE_FORWARD' | 'TURN_LEFT' | 'TURN_RIGHT' | 'STOP';

type ModelPayload = {
  architecture: number[];
  image_size: number;
  actions: ActionName[];
  layers: Array<{ weight: number[][]; bias: number[] }>;
  metrics: {
    dataset_samples: number;
    train_accuracy: number;
    validation_accuracy: number;
    test_accuracy: number;
    linear_baseline_test_accuracy: number;
  };
};

type MetricSummary = { mean: number; std: number };
type HW3Summary = {
  experiment: string;
  controls: { seeds: number[]; epochs: number; split: Record<'train' | 'val' | 'test', number> };
  models: Record<
    'baseline' | 'gated_residual' | 'ungated_residual',
    {
      label: string;
      parameter_count: number;
      test_accuracy: MetricSummary;
      test_macro_f1: MetricSummary;
      test_loss: MetricSummary;
      per_class_accuracy: Record<ActionName, MetricSummary>;
      gate_statistics?: { mean: MetricSummary; saturation_fraction: MetricSummary };
    }
  >;
};

type FrameAnnotation = {
  id: string;
  frame: number;
  label: FrameLabel;
  tool: Exclude<DrawTool, 'inspect'>;
  x: number;
  y: number;
  width?: number;
  height?: number;
};

type SegmentAnnotation = {
  id: string;
  start: number;
  end: number;
  label: SegmentLabel;
};

const frameLabels: Array<{ value: FrameLabel; label: string; color: string }> = [
  { value: 'landmark', label: 'Landmark', color: '#d7f99b' },
  { value: 'obstacle', label: 'Obstacle', color: '#ff9b77' },
  { value: 'navigable', label: 'Navigable area', color: '#7ee2c3' },
  { value: 'localization-cue', label: 'Localization cue', color: '#8ec5ff' },
  { value: 'goal-evidence', label: 'Goal evidence', color: '#f4d96f' },
  { value: 'ambiguous', label: 'Ambiguous region', color: '#c9b1ff' },
];

const segmentLabels: Array<{ value: SegmentLabel; label: string; color: string }> = [
  { value: 'exploration', label: 'Exploration', color: '#8aa39a' },
  { value: 'correct-progress', label: 'Correct progress', color: '#61a77c' },
  { value: 'wrong-turn', label: 'Wrong turn', color: '#e09063' },
  { value: 'collision-recovery', label: 'Collision recovery', color: '#d66565' },
  { value: 'goal-recognition', label: 'Goal recognition', color: '#d2b84f' },
];

const frames = Array.from({ length: TOTAL_STEPS }, (_, step) => ({
  step,
  image: framePath(step),
  depth: depthPath(step),
  map: mapPath(step),
  time: step / SAMPLE_RATE,
  action:
    step === TOTAL_STEPS - 1
      ? 'STOP'
      : step < 4 || (step >= 12 && step < 17) || step > 23
        ? 'MOVE_FORWARD'
        : step < 7 || (step >= 18 && step < 21)
          ? 'TURN_LEFT'
          : 'TURN_RIGHT',
  collision: step === 9 || step === 21,
  distance: Math.max(0.35, 8.8 - step * 0.285),
}));

const initialSegments: SegmentAnnotation[] = [
  { id: 'seed-a', start: 0, end: 6, label: 'exploration' },
  { id: 'seed-b', start: 7, end: 18, label: 'correct-progress' },
  { id: 'seed-c', start: 19, end: 22, label: 'collision-recovery' },
  { id: 'seed-d', start: 23, end: 29, label: 'goal-recognition' },
];

const clamp = (value: number, min: number, max: number) =>
  Math.min(max, Math.max(min, value));

const labelForFrame = (value: FrameLabel) =>
  frameLabels.find((item) => item.value === value) ?? frameLabels[0];
const labelForSegment = (value: SegmentLabel) =>
  segmentLabels.find((item) => item.value === value) ?? segmentLabels[0];
const stepPercent = (step: number) => (step / (TOTAL_STEPS - 1)) * 100;

const actionLabels: Record<ActionName, string> = {
  MOVE_FORWARD: 'Move forward',
  TURN_LEFT: 'Turn left',
  TURN_RIGHT: 'Turn right',
  STOP: 'Stop',
};

const predictWithModel = (model: ModelPayload, features: number[]) => {
  let values = features;
  model.layers.forEach((layer, layerIndex) => {
    values = layer.weight.map((row, outputIndex) => {
      const value = row.reduce((sum, weight, inputIndex) => sum + weight * values[inputIndex], layer.bias[outputIndex]);
      return layerIndex < model.layers.length - 1 ? Math.max(0, value) : value;
    });
  });
  const maximum = Math.max(...values);
  const exponentials = values.map((value) => Math.exp(value - maximum));
  const total = exponentials.reduce((sum, value) => sum + value, 0);
  return exponentials.map((value) => value / total);
};

export default function Home() {
  const [currentStep, setCurrentStep] = useState(14);
  const [activeTab, setActiveTab] = useState<'frame' | 'segment' | 'model'>('model');
  const [drawTool, setDrawTool] = useState<DrawTool>('inspect');
  const [frameLabel, setFrameLabel] = useState<FrameLabel>('landmark');
  const [frameAnnotations, setFrameAnnotations] = useState<FrameAnnotation[]>([]);
  const [segmentLabel, setSegmentLabel] = useState<SegmentLabel>('correct-progress');
  const [rangeStart, setRangeStart] = useState(7);
  const [rangeEnd, setRangeEnd] = useState(18);
  const [segments, setSegments] = useState<SegmentAnnotation[]>(initialSegments);
  const [thumbnailWidth, setThumbnailWidth] = useState(104);
  const [isPlaying, setIsPlaying] = useState(false);
  const [goalDistance, setGoalDistance] = useState(4.5);
  const [goalAngle, setGoalAngle] = useState(0);
  const [model, setModel] = useState<ModelPayload | null>(null);
  const [hw3Summary, setHW3Summary] = useState<HW3Summary | null>(null);
  const [prediction, setPrediction] = useState<{ action: ActionName; probabilities: number[] } | null>(null);
  const [isPredicting, setIsPredicting] = useState(false);
  const [draftBox, setDraftBox] = useState<{
    startX: number;
    startY: number;
    x: number;
    y: number;
    width: number;
    height: number;
  } | null>(null);
  const [notice, setNotice] = useState('Session annotations are ready to export.');
  const filmstripRef = useRef<HTMLDivElement>(null);
  const importRef = useRef<HTMLInputElement>(null);

  const current = frames[currentStep];
  const currentAnnotations = frameAnnotations.filter((item) => item.frame === currentStep);
  const progress = `${stepPercent(currentStep)}%`;

  const activeSegments = useMemo(
    () => segments.filter((item) => currentStep >= item.start && currentStep <= item.end),
    [currentStep, segments],
  );

  useEffect(() => {
    void fetch('/model/navigation_mlp.json')
      .then(async (response) => (await response.json()) as ModelPayload)
      .then((payload) => setModel(payload))
      .catch(() => setNotice('Could not load the trained navigation model.'));
  }, []);

  useEffect(() => {
    void fetch('/model/hw3_summary.json')
      .then(async (response) => (await response.json()) as HW3Summary)
      .then((payload) => setHW3Summary(payload))
      .catch(() => setNotice('Could not load the HW3 experiment summary.'));
  }, []);

  useEffect(() => {
    // oxlint-disable-next-line react-compiler
    setPrediction(null);
  }, [currentStep, goalDistance, goalAngle]);

  useEffect(() => {
    if (!isPlaying) return;
    const timer = window.setInterval(() => {
      setCurrentStep((step) => {
        if (step >= TOTAL_STEPS - 1) {
          setIsPlaying(false);
          return step;
        }
        return step + 1;
      });
    }, 600);
    return () => window.clearInterval(timer);
  }, [isPlaying]);

  useEffect(() => {
    const selected = filmstripRef.current?.querySelector<HTMLElement>(
      `[data-step="${currentStep}"]`,
    );
    selected?.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' });
  }, [currentStep]);

  useEffect(() => {
    const handleKey = (event: KeyboardEvent) => {
      if (event.key === 'ArrowLeft') setCurrentStep((step) => Math.max(0, step - 1));
      if (event.key === 'ArrowRight') setCurrentStep((step) => Math.min(TOTAL_STEPS - 1, step + 1));
    };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, []);

  const pointFromEvent = (event: ReactPointerEvent<HTMLDivElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    return {
      x: clamp(((event.clientX - rect.left) / rect.width) * 100, 0, 100),
      y: clamp(((event.clientY - rect.top) / rect.height) * 100, 0, 100),
    };
  };

  const addPoint = (x: number, y: number) => {
    setFrameAnnotations((items) => [
      ...items,
      { id: crypto.randomUUID(), frame: currentStep, label: frameLabel, tool: 'point', x, y },
    ]);
    setNotice(`Added ${labelForFrame(frameLabel).label} to frame ${currentStep + 1}.`);
  };

  const onCanvasPointerDown = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (drawTool === 'inspect') return;
    event.currentTarget.setPointerCapture(event.pointerId);
    const point = pointFromEvent(event);
    if (drawTool === 'point') {
      addPoint(point.x, point.y);
      return;
    }
    setDraftBox({ startX: point.x, startY: point.y, x: point.x, y: point.y, width: 0, height: 0 });
  };

  const onCanvasPointerMove = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (!draftBox || drawTool !== 'box') return;
    const point = pointFromEvent(event);
    setDraftBox((draft) =>
      draft
        ? {
            ...draft,
            x: Math.min(draft.startX, point.x),
            y: Math.min(draft.startY, point.y),
            width: Math.abs(point.x - draft.startX),
            height: Math.abs(point.y - draft.startY),
          }
        : null,
    );
  };

  const onCanvasPointerUp = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (!draftBox || drawTool !== 'box') return;
    event.currentTarget.releasePointerCapture(event.pointerId);
    if (draftBox.width > 2 && draftBox.height > 2) {
      setFrameAnnotations((items) => [
        ...items,
        {
          id: crypto.randomUUID(),
          frame: currentStep,
          label: frameLabel,
          tool: 'box',
          x: draftBox.x,
          y: draftBox.y,
          width: draftBox.width,
          height: draftBox.height,
        },
      ]);
      setNotice(`Added ${labelForFrame(frameLabel).label} box to frame ${currentStep + 1}.`);
    }
    setDraftBox(null);
  };

  const addSegment = () => {
    const start = Math.min(rangeStart, rangeEnd);
    const end = Math.max(rangeStart, rangeEnd);
    setSegments((items) => [
      ...items,
      { id: crypto.randomUUID(), start, end, label: segmentLabel },
    ]);
    setNotice(`Added ${labelForSegment(segmentLabel).label}: ${start + 1}–${end + 1}.`);
  };

  const exportAnnotations = () => {
    const payload = {
      schema_version: '1.0',
      dataset: 'Habitat-Lab official PointNav visualization sample',
      trajectory_id: 'skokloster-pointnav-demo-001',
      provenance: {
        source: 'https://github.com/facebookresearch/habitat-lab',
        asset:
          'docs/images/habitat-lab-tdmap-viz-images/skokloster-castle.glb_3662.gif',
        note: 'RGB/depth/map frames are official. Actions, distances and timing shown in the UI are illustrative metadata for the annotation prototype.',
      },
      frame_annotations: frameAnnotations,
      segment_annotations: segments,
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'habitat-trajectory-annotations.json';
    link.click();
    URL.revokeObjectURL(url);
    setNotice('Annotation JSON exported.');
  };

  const importAnnotations = async (file: File | undefined) => {
    if (!file) return;
    try {
      const payload = JSON.parse(await file.text());
      if (Array.isArray(payload.frame_annotations)) setFrameAnnotations(payload.frame_annotations);
      if (Array.isArray(payload.segment_annotations)) setSegments(payload.segment_annotations);
      setNotice(`Imported annotations from ${file.name}.`);
    } catch {
      setNotice('Could not import this JSON file.');
    }
    if (importRef.current) importRef.current.value = '';
  };

  const runPrediction = async () => {
    if (!model) return;
    setIsPredicting(true);
    try {
      const image = new window.Image();
      image.src = current.image;
      await image.decode();
      const canvas = document.createElement('canvas');
      canvas.width = model.image_size;
      canvas.height = model.image_size;
      const context = canvas.getContext('2d', { willReadFrequently: true });
      if (!context) throw new Error('Canvas is unavailable');
      context.drawImage(image, 0, 0, model.image_size, model.image_size);
      const rgba = context.getImageData(0, 0, model.image_size, model.image_size).data;
      const features: number[] = [];
      for (let index = 0; index < rgba.length; index += 4) {
        features.push(rgba[index] / 255, rgba[index + 1] / 255, rgba[index + 2] / 255);
      }
      features.push(goalDistance / 10, (goalAngle * Math.PI) / 180 / Math.PI);
      const probabilities = predictWithModel(model, features);
      const bestIndex = probabilities.indexOf(Math.max(...probabilities));
      setPrediction({ action: model.actions[bestIndex], probabilities });
      setNotice(`Model predicted ${model.actions[bestIndex]} for frame ${currentStep + 1}.`);
    } catch {
      setNotice('Prediction failed. Please try another frame.');
    } finally {
      setIsPredicting(false);
    }
  };

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand-lockup">
          <span className="brand-mark"><Route size={17} /></span>
          <div><p>Habitat navigation learner</p><span>Trajectory annotation + neural design experiment</span></div>
        </div>
        <div className="episode-meta">
          <Badge variant="outline">HW3 · POINTNAV</Badge>
          <span>Skokloster Castle</span>
          <span>{TOTAL_STEPS} sampled observations</span>
        </div>
        <div className="header-actions">
          <input ref={importRef} hidden type="file" accept="application/json" onChange={(event) => void importAnnotations(event.target.files?.[0])} />
          <Button size="sm" variant="ghost" onClick={() => importRef.current?.click()}><Upload /> Import</Button>
          <Button size="sm" variant="outline" onClick={exportAnnotations}><Download /> Export JSON</Button>
        </div>
      </header>

      <section className="workspace-grid">
        <aside className="trajectory-sidebar panel">
          <div className="panel-heading">
            <div><span className="kicker">Dataset</span><h2>Trajectories</h2></div>
            <Badge>1 sample</Badge>
          </div>
          <button className="trajectory-card active" type="button">
            <div className="trajectory-card__top"><strong>PointNav · 001</strong><span>active</span></div>
            <p>RGB + depth + synchronized top-down map</p>
            <div className="mini-progress"><span style={{ width: progress }} /></div>
            <small>Frame {currentStep + 1} / {TOTAL_STEPS}</small>
          </button>
          <dl className="trajectory-stats">
            <div><dt>Task</dt><dd>PointGoal navigation</dd></div>
            <div><dt>Sample rate</dt><dd>{SAMPLE_RATE} fps</dd></div>
            <div><dt>Frame labels</dt><dd>{frameAnnotations.length}</dd></div>
            <div><dt>Segments</dt><dd>{segments.length}</dd></div>
          </dl>
          <div className="source-note"><ScanLine size={16} /><p>Images are sampled from an official Habitat-Lab PointNav visualization. UI action and distance fields are illustrative.</p></div>
        </aside>

        <section className="observation-panel panel">
          <div className="panel-heading compact">
            <div><span className="kicker">Selected observation</span><h2>RGB · frame {String(currentStep + 1).padStart(3, '0')}</h2></div>
            <div className="frame-nav">
              <Button size="icon-sm" variant="outline" aria-label="Previous frame" onClick={() => setCurrentStep((value) => Math.max(0, value - 1))}><ChevronLeft /></Button>
              <Button size="icon-sm" variant="outline" aria-label={isPlaying ? 'Pause' : 'Play'} onClick={() => setIsPlaying((value) => !value)}>{isPlaying ? <Pause /> : <Play />}</Button>
              <Button size="icon-sm" variant="outline" aria-label="Next frame" onClick={() => setCurrentStep((value) => Math.min(TOTAL_STEPS - 1, value + 1))}><ChevronRight /></Button>
            </div>
          </div>
          <div className="observation-stage">
            <div
              className={`image-canvas tool-${drawTool}`}
              onPointerDown={onCanvasPointerDown}
              onPointerMove={onCanvasPointerMove}
              onPointerUp={onCanvasPointerUp}
            >
              <img src={current.image} alt={`Habitat RGB observation at frame ${currentStep + 1}`} draggable={false} />
              {currentAnnotations.map((item) => {
                const definition = labelForFrame(item.label);
                return item.tool === 'point' ? (
                  <button
                    key={item.id}
                    type="button"
                    className="annotation-point"
                    style={{ left: `${item.x}%`, top: `${item.y}%`, '--annotation-color': definition.color } as React.CSSProperties}
                    title={`${definition.label} — click to remove`}
                    onPointerDown={(event) => event.stopPropagation()}
                    onClick={() => setFrameAnnotations((items) => items.filter((entry) => entry.id !== item.id))}
                  ><span>{definition.label.slice(0, 1)}</span></button>
                ) : (
                  <button
                    key={item.id}
                    type="button"
                    className="annotation-box"
                    style={{ left: `${item.x}%`, top: `${item.y}%`, width: `${item.width}%`, height: `${item.height}%`, '--annotation-color': definition.color } as React.CSSProperties}
                    title={`${definition.label} — click to remove`}
                    onPointerDown={(event) => event.stopPropagation()}
                    onClick={() => setFrameAnnotations((items) => items.filter((entry) => entry.id !== item.id))}
                  ><span>{definition.label}</span></button>
                );
              })}
              {draftBox && <div className="annotation-box draft" style={{ left: `${draftBox.x}%`, top: `${draftBox.y}%`, width: `${draftBox.width}%`, height: `${draftBox.height}%`, '--annotation-color': labelForFrame(frameLabel).color } as React.CSSProperties} />}
            </div>
            <span className="frame-chip">t = {current.time.toFixed(1)}s</span>
          </div>
          <div className="observation-metadata">
            <div><span>Action</span><strong>{current.action}</strong></div>
            <div><span>Goal distance</span><strong>{current.distance.toFixed(2)} m</strong></div>
            <div><span>Collision</span><strong className={current.collision ? 'danger' : ''}>{current.collision ? 'TRUE' : 'FALSE'}</strong></div>
            <div><span>Active segment</span><strong>{activeSegments[0] ? labelForSegment(activeSegments[0].label).label : 'Unlabeled'}</strong></div>
          </div>
        </section>

        <aside className="annotation-panel panel">
          <div className="annotation-tabs" role="tablist">
            <button type="button" className={activeTab === 'frame' ? 'active' : ''} onClick={() => setActiveTab('frame')}>Frame annotation</button>
            <button type="button" className={activeTab === 'segment' ? 'active' : ''} onClick={() => setActiveTab('segment')}>Trajectory segment</button>
            <button type="button" className={activeTab === 'model' ? 'active' : ''} onClick={() => setActiveTab('model')}>MLP prediction</button>
          </div>
          {activeTab === 'model' ? (
            <div className="annotation-form prediction-form">
              <div className="model-summary"><BrainCircuit /><div><span className="kicker">Trained model</span><strong>Navigation MLP</strong><small>194 → 64 → 32 → 4</small></div></div>
              <label className="slider-field"><span><b>Goal distance</b><output>{goalDistance.toFixed(1)} m</output></span><input aria-label="Goal distance" type="range" min="0.1" max="10" step="0.1" value={goalDistance} onChange={(event) => setGoalDistance(Number(event.target.value))} /></label>
              <label className="slider-field"><span><b>Relative goal angle</b><output>{goalAngle > 0 ? '+' : ''}{goalAngle}°</output></span><input aria-label="Relative goal angle" type="range" min="-180" max="180" step="5" value={goalAngle} onChange={(event) => setGoalAngle(Number(event.target.value))} /></label>
              <div className="angle-hint"><span>← goal left</span><i style={{ left: `${((goalAngle + 180) / 360) * 100}%` }} /><span>goal right →</span></div>
              <Button className="w-full predict-button" disabled={!model || isPredicting} onClick={() => void runPrediction()}><Sparkles /> {isPredicting ? 'Running model…' : 'Predict next action'}</Button>
              {prediction ? (
                <div className="prediction-result">
                  <span className="kicker">Predicted action</span>
                  <strong>{actionLabels[prediction.action]}</strong>
                  <small>{(Math.max(...prediction.probabilities) * 100).toFixed(1)}% confidence</small>
                  <div className="probability-list">{model?.actions.map((action, index) => <div key={action}><span>{actionLabels[action]}</span><i><b style={{ width: `${prediction.probabilities[index] * 100}%` }} /></i><output>{(prediction.probabilities[index] * 100).toFixed(0)}%</output></div>)}</div>
                </div>
              ) : <div className="prediction-placeholder"><Sparkles /><p>Select a frame, set the relative goal, and run the trained model entirely in the browser.</p></div>}
            </div>
          ) : activeTab === 'frame' ? (
            <div className="annotation-form">
              <div><span className="kicker">Geometry</span><div className="tool-grid">
                <button type="button" className={drawTool === 'inspect' ? 'active' : ''} onClick={() => setDrawTool('inspect')}><MousePointer2 /> Inspect</button>
                <button type="button" className={drawTool === 'point' ? 'active' : ''} onClick={() => setDrawTool('point')}><CircleDot /> Point</button>
                <button type="button" className={drawTool === 'box' ? 'active' : ''} onClick={() => setDrawTool('box')}><BoxSelect /> Box</button>
              </div></div>
              <label className="field-label"><span>Frame label</span><select value={frameLabel} onChange={(event) => setFrameLabel(event.target.value as FrameLabel)}>{frameLabels.map((item) => <option key={item.value} value={item.value}>{item.label}</option>)}</select></label>
              <div className="instruction-card"><Info /><p>{drawTool === 'inspect' ? 'Choose Point or Box, then annotate the selected RGB observation.' : drawTool === 'point' ? 'Click a spatial cue in the image. Click an existing point to remove it.' : 'Drag a box around a region. Click an existing box to remove it.'}</p></div>
              <div className="annotation-list-header"><span>On this frame</span><Badge variant="outline">{currentAnnotations.length}</Badge></div>
              <div className="annotation-list">
                {currentAnnotations.length === 0 ? <p className="empty-state">No annotations on frame {currentStep + 1}.</p> : currentAnnotations.map((item) => {
                  const definition = labelForFrame(item.label);
                  return <div key={item.id} className="annotation-row"><i style={{ background: definition.color }} /><span><strong>{definition.label}</strong><small>{item.tool} · x {item.x.toFixed(1)} · y {item.y.toFixed(1)}</small></span><button type="button" aria-label="Delete annotation" onClick={() => setFrameAnnotations((items) => items.filter((entry) => entry.id !== item.id))}><Trash2 /></button></div>;
                })}
              </div>
            </div>
          ) : (
            <div className="annotation-form">
              <label className="field-label"><span>Segment label</span><select value={segmentLabel} onChange={(event) => setSegmentLabel(event.target.value as SegmentLabel)}>{segmentLabels.map((item) => <option key={item.value} value={item.value}>{item.label}</option>)}</select></label>
              <div className="range-grid">
                <label><span>Start frame</span><input type="number" min={1} max={TOTAL_STEPS} value={rangeStart + 1} onChange={(event) => setRangeStart(clamp(Number(event.target.value) - 1, 0, TOTAL_STEPS - 1))} /><button type="button" onClick={() => setRangeStart(currentStep)}>Use current</button></label>
                <label><span>End frame</span><input type="number" min={1} max={TOTAL_STEPS} value={rangeEnd + 1} onChange={(event) => setRangeEnd(clamp(Number(event.target.value) - 1, 0, TOTAL_STEPS - 1))} /><button type="button" onClick={() => setRangeEnd(currentStep)}>Use current</button></label>
              </div>
              <Button className="w-full" onClick={addSegment}><Flag /> Add labeled interval</Button>
              <Button className="w-full" variant="outline" onClick={() => { setRangeStart(0); setRangeEnd(TOTAL_STEPS - 1); }}>Select entire trajectory</Button>
              <div className="annotation-list-header"><span>Trajectory labels</span><Badge variant="outline">{segments.length}</Badge></div>
              <div className="annotation-list segment-list">
                {segments.map((item) => {
                  const definition = labelForSegment(item.label);
                  return <div key={item.id} className="annotation-row"><i style={{ background: definition.color }} /><span><strong>{definition.label}</strong><small>frames {item.start + 1}–{item.end + 1}</small></span><button type="button" aria-label="Delete segment" onClick={() => setSegments((items) => items.filter((entry) => entry.id !== item.id))}><Trash2 /></button></div>;
                })}
              </div>
            </div>
          )}
        </aside>
      </section>

      <section className="timeline-panel panel">
        <div className="timeline-title-row">
          <div><span className="kicker">Long horizontal data</span><h2>Overview + detail timeline</h2></div>
          <div className="timeline-controls"><span>Thumbnail width</span><input type="range" min="76" max="160" value={thumbnailWidth} onChange={(event) => setThumbnailWidth(Number(event.target.value))} /></div>
        </div>
        <div className="overview-section">
          <span className="track-label">OVERVIEW</span>
          <button type="button" className="overview-track" aria-label="Jump to a trajectory frame" onClick={(event) => {
            const rect = event.currentTarget.getBoundingClientRect();
            setCurrentStep(Math.round(((event.clientX - rect.left) / rect.width) * (TOTAL_STEPS - 1)));
          }}>
            {segments.map((item) => {
              const definition = labelForSegment(item.label);
              return <span key={item.id} className="segment-span" style={{ left: `${stepPercent(item.start)}%`, width: `${Math.max(1.2, stepPercent(item.end) - stepPercent(item.start))}%`, background: definition.color }} title={`${definition.label}: ${item.start + 1}–${item.end + 1}`} />;
            })}
            <i className="playhead" style={{ left: progress }} />
          </button>
        </div>
        <div className="detail-row"><span className="track-label">RGB</span><div ref={filmstripRef} className="filmstrip" aria-label="Trajectory frames">
          {frames.map((frame) => {
            const count = frameAnnotations.filter((item) => item.frame === frame.step).length;
            return <button key={frame.step} data-step={frame.step} type="button" className={frame.step === currentStep ? 'selected' : ''} style={{ minWidth: thumbnailWidth }} onClick={() => setCurrentStep(frame.step)}><img src={frame.image} alt="" /><span>{String(frame.step + 1).padStart(3, '0')}</span>{count > 0 && <b>{count}</b>}</button>;
          })}
        </div></div>
        <div className="event-track-row"><span className="track-label">ACTION</span><div className="event-track">{frames.map((frame) => <button key={frame.step} type="button" aria-label={`Frame ${frame.step + 1}: ${frame.action}`} style={{ width: `${100 / TOTAL_STEPS}%` }} className={`action-cell ${frame.action.toLowerCase()} ${frame.step === currentStep ? 'selected' : ''}`} title={`${frame.step + 1}: ${frame.action}`} onClick={() => setCurrentStep(frame.step)} />)}</div></div>
        <div className="event-track-row"><span className="track-label">EVENTS</span><div className="event-track">{frames.map((frame) => <button key={frame.step} type="button" aria-label={frame.collision ? `Collision at frame ${frame.step + 1}` : `No event at frame ${frame.step + 1}`} style={{ width: `${100 / TOTAL_STEPS}%` }} className={`event-cell ${frame.collision ? 'collision' : ''}`} title={frame.collision ? `Collision at frame ${frame.step + 1}` : `Frame ${frame.step + 1}`} onClick={() => setCurrentStep(frame.step)} />)}</div></div>
        <div className="timeline-legend"><span><i className="legend-forward" /> forward</span><span><i className="legend-left" /> turn left</span><span><i className="legend-right" /> turn right</span><span><i className="legend-collision" /> collision</span><p>{notice}</p></div>
      </section>

      <section className="model-monitor panel">
        <div className="monitor-copy"><span className="kicker">HW2 learning pipeline</span><h2>RGB + relative goal → next action</h2><p>The selected Habitat RGB frame is resized to 8×8 RGB values and concatenated with normalized goal distance and angle. Two ReLU hidden layers produce four action probabilities.</p><div className="pipeline"><span>194 inputs</span><i>→</i><span>64 ReLU</span><i>→</i><span>32 ReLU</span><i>→</i><span>4 actions</span></div></div>
        <div className="training-plot"><img src="/model/training_curves.png" alt="Training and validation loss and accuracy curves over 90 epochs" /></div>
        <div className="metric-stack"><span className="kicker">Held-out evaluation</span><div><strong>{model ? `${(model.metrics.test_accuracy * 100).toFixed(1)}%` : '—'}</strong><small>MLP test accuracy</small></div><div><strong>{model ? `${(model.metrics.linear_baseline_test_accuracy * 100).toFixed(1)}%` : '—'}</strong><small>Linear baseline</small></div><div><strong>{model?.metrics.dataset_samples ?? '—'}</strong><small>Emulated samples</small></div><p>Split by episode: 70% train · 15% validation · 15% test.</p></div>
      </section>

      <section className="hw3-lab panel">
        <div className="hw3-heading">
          <div>
            <span className="kicker">HW3 · Break the network</span>
            <h2>Goal-conditioned gated residual fusion</h2>
            <p>The HW2 baseline mixes 192 RGB values with only two goal values at the first layer. The proposed model gives each modality its own encoder, then lets the goal vector control a residual correction to the visual representation.</p>
          </div>
          <div className="hypothesis-card"><Sparkles /><p><strong>Hypothesis</strong>The gate should make goal context harder to ignore while preserving a direct visual path. It may fail if the sigmoid saturates and becomes an almost-always-open switch.</p></div>
        </div>

        <div className="architecture-flow" aria-label="Goal-gated residual architecture diagram">
          <div><span>RGB 8×8×3</span><strong>Visual encoder</strong><small>192 → 64 ReLU</small></div>
          <i>+</i>
          <div><span>Goal (d, θ)</span><strong>Goal encoder</strong><small>2 → 16 ReLU</small></div>
          <i>→</i>
          <div className="architecture-accent"><span>Residual fusion</span><strong>v + σ(Wg) ⊙ tanh(Wc)</strong><small>goal-controlled correction</small></div>
          <i>→</i>
          <div><span>Classifier</span><strong>64 → 32 → 4</strong><small>next action logits</small></div>
        </div>

        <div className="experiment-grid">
          {(['baseline', 'gated_residual', 'ungated_residual'] as const).map((key) => {
            const result = hw3Summary?.models[key];
            return (
              <article key={key} className={key === 'gated_residual' ? 'featured' : ''}>
                <div className="experiment-card-title"><span>{key === 'baseline' ? 'Control' : key === 'gated_residual' ? 'Proposed' : 'Ablation'}</span>{key === 'gated_residual' && <Badge>Full design</Badge>}</div>
                <h3>{result?.label ?? (key === 'baseline' ? 'HW2 baseline' : key === 'gated_residual' ? 'Goal-gated residual' : 'No-gate ablation')}</h3>
                <strong>{result ? `${(result.test_accuracy.mean * 100).toFixed(2)}%` : '—'}</strong>
                <small>mean test accuracy ± {result ? `${(result.test_accuracy.std * 100).toFixed(2)}` : '—'} pp</small>
                <dl><div><dt>Macro F1</dt><dd>{result ? (result.test_macro_f1.mean * 100).toFixed(2) : '—'}%</dd></div><div><dt>Parameters</dt><dd>{result?.parameter_count.toLocaleString() ?? '—'}</dd></div><div><dt>Forward accuracy</dt><dd>{result ? (result.per_class_accuracy.MOVE_FORWARD.mean * 100).toFixed(1) : '—'}%</dd></div></dl>
              </article>
            );
          })}
        </div>

        <div className="evidence-grid">
          <figure><img src="/model/hw3_training_curves.png" alt="Validation curves for baseline, gated residual, and no-gate ablation over five seeds" /><figcaption>Validation curves · mean ± one standard deviation across five seeds</figcaption></figure>
          <figure><img src="/model/hw3_confusion_matrices.png" alt="Mean test confusion matrices for the three HW3 models" /><figcaption>Per-class evidence · row-normalized test confusion matrices</figcaption></figure>
        </div>

        <div className="finding-strip">
          <BrainCircuit />
          <p><strong>What happened:</strong> the proposed model improved mean test accuracy by {hw3Summary ? `${((hw3Summary.models.gated_residual.test_accuracy.mean - hw3Summary.models.baseline.test_accuracy.mean) * 100).toFixed(2)}` : '—'} percentage points over HW2. The no-gate ablation captured most of that gain, while the learned gate added only {hw3Summary ? `${((hw3Summary.models.gated_residual.test_accuracy.mean - hw3Summary.models.ungated_residual.test_accuracy.mean) * 100).toFixed(2)}` : '—'} points. Its mean activation was {hw3Summary?.models.gated_residual.gate_statistics ? hw3Summary.models.gated_residual.gate_statistics.mean.mean.toFixed(2) : '—'}, with {hw3Summary?.models.gated_residual.gate_statistics ? `${(hw3Summary.models.gated_residual.gate_statistics.saturation_fraction.mean * 100).toFixed(1)}%` : '—'} of gates near saturation—evidence that separate modality processing and the residual path mattered more than fine-grained gating.</p>
        </div>
      </section>

      <footer className="provenance-bar"><Map /><p><strong>Public data source:</strong> Habitat-Lab official PointNav top-down-map visualization. The original GIF provides RGB, depth and map imagery; this prototype samples 30 synchronized observations for interface design.</p><a href="https://github.com/facebookresearch/habitat-lab" target="_blank" rel="noreferrer">View source</a></footer>
    </main>
  );
}

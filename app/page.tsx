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

export default function Home() {
  const [currentStep, setCurrentStep] = useState(14);
  const [activeTab, setActiveTab] = useState<'frame' | 'segment'>('frame');
  const [drawTool, setDrawTool] = useState<DrawTool>('inspect');
  const [frameLabel, setFrameLabel] = useState<FrameLabel>('landmark');
  const [frameAnnotations, setFrameAnnotations] = useState<FrameAnnotation[]>([]);
  const [segmentLabel, setSegmentLabel] = useState<SegmentLabel>('correct-progress');
  const [rangeStart, setRangeStart] = useState(7);
  const [rangeEnd, setRangeEnd] = useState(18);
  const [segments, setSegments] = useState<SegmentAnnotation[]>(initialSegments);
  const [thumbnailWidth, setThumbnailWidth] = useState(104);
  const [isPlaying, setIsPlaying] = useState(false);
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

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand-lockup">
          <span className="brand-mark"><Route size={17} /></span>
          <div><p>Habitat trajectory annotator</p><span>Spatial intelligence data workbench</span></div>
        </div>
        <div className="episode-meta">
          <Badge variant="outline">POINTNAV · 001</Badge>
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
          </div>
          {activeTab === 'frame' ? (
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

      <footer className="provenance-bar"><Map /><p><strong>Public data source:</strong> Habitat-Lab official PointNav top-down-map visualization. The original GIF provides RGB, depth and map imagery; this prototype samples 30 synchronized observations for interface design.</p><a href="https://github.com/facebookresearch/habitat-lab" target="_blank" rel="noreferrer">View source</a></footer>
    </main>
  );
}

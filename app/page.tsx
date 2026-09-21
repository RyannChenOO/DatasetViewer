'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import {
  ChevronLeft,
  ChevronRight,
  CircleDot,
  Compass,
  Download,
  FileImage,
  Flag,
  Focus,
  ImagePlus,
  Info,
  Landmark,
  MapPinned,
  Minus,
  MousePointer2,
  Plus,
  RotateCcw,
  Route,
  ScanSearch,
  Trash2,
  TriangleAlert,
} from 'lucide-react';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';

type Scene = {
  id: string;
  title: string;
  subtitle: string;
  image: string;
  map: string;
  view: string;
  step: string;
  distance: string;
  heading: string;
  sceneType: string;
  instruction: string;
  source?: 'sample' | 'upload';
};

type AnnotationType = 'landmark' | 'path' | 'obstacle' | 'goal';

type Annotation = {
  id: string;
  type: AnnotationType;
  x: number;
  y: number;
};

const sampleScenes: Scene[] = [
  {
    id: 'E-042-front',
    title: 'North Walk',
    subtitle: 'Front observation',
    image: '/data/episode-01.jpg',
    map: '/data/episode-03.jpg',
    view: 'Front view',
    step: '08',
    distance: '18.4 m',
    heading: 'NE 42°',
    sceneType: 'Building',
    instruction: 'Follow the brick path, keep the building on your left, then turn toward the courtyard.',
    source: 'sample',
  },
  {
    id: 'E-042-right',
    title: 'Hall Edge',
    subtitle: 'Right observation',
    image: '/data/episode-02.jpg',
    map: '/data/episode-03.jpg',
    view: 'Right view',
    step: '08',
    distance: '18.4 m',
    heading: 'E 87°',
    sceneType: 'Building',
    instruction: 'Use the hall facade and lamp posts as stable landmarks while rotating toward the goal.',
    source: 'sample',
  },
  {
    id: 'E-042-goal',
    title: 'Brick Path Goal',
    subtitle: 'Goal image',
    image: '/data/episode-05.jpg',
    map: '/data/episode-04.jpg',
    view: 'Goal image',
    step: 'Goal',
    distance: '0.0 m',
    heading: 'E 90°',
    sceneType: 'Building',
    instruction: 'Match the path alignment and building edge in this goal observation.',
    source: 'sample',
  },
  {
    id: 'E-114-failure',
    title: 'Foliage Occlusion',
    subtitle: 'Failure case',
    image: '/data/episode-11.jpg',
    map: '/data/episode-04.jpg',
    view: 'Front view',
    step: '31',
    distance: '11.7 m',
    heading: 'SW 214°',
    sceneType: 'Intersection',
    instruction: 'Recover from visual occlusion by tracking the last reliable path direction.',
    source: 'sample',
  },
];

const annotationTypes: Array<{
  id: AnnotationType;
  label: string;
  color: string;
  icon: typeof Landmark;
}> = [
  { id: 'landmark', label: 'Landmark', color: '#b8f36a', icon: Landmark },
  { id: 'path', label: 'Path', color: '#66d9c4', icon: Route },
  { id: 'obstacle', label: 'Obstacle', color: '#ff9f68', icon: TriangleAlert },
  { id: 'goal', label: 'Goal', color: '#e9db75', icon: Flag },
];

const clamp = (value: number, min: number, max: number) =>
  Math.min(max, Math.max(min, value));

export default function Home() {
  const [scenes, setScenes] = useState(sampleScenes);
  const [selectedId, setSelectedId] = useState(sampleScenes[0].id);
  const [zoom, setZoom] = useState(100);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const [annotationMode, setAnnotationMode] = useState(false);
  const [annotationType, setAnnotationType] = useState<AnnotationType>('landmark');
  const [annotations, setAnnotations] = useState<Record<string, Annotation[]>>({});
  const [annotationsReady, setAnnotationsReady] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const dragRef = useRef<{
    active: boolean;
    startX: number;
    startY: number;
    originX: number;
    originY: number;
  }>({ active: false, startX: 0, startY: 0, originX: 0, originY: 0 });

  const scene = useMemo(
    () => scenes.find((item) => item.id === selectedId) ?? scenes[0],
    [scenes, selectedId],
  );
  const sceneAnnotations = annotations[scene.id] ?? [];
  const selectedIndex = scenes.findIndex((item) => item.id === scene.id);

  useEffect(() => {
    try {
      const stored = localStorage.getItem('locus-annotations');
      if (stored) setAnnotations(JSON.parse(stored));
    } catch {
      // Keep the viewer usable if browser storage is unavailable.
    } finally {
      setAnnotationsReady(true);
    }
  }, []);

  useEffect(() => {
    if (!annotationsReady) return;
    localStorage.setItem('locus-annotations', JSON.stringify(annotations));
  }, [annotations, annotationsReady]);

  useEffect(() => {
    return () => {
      scenes
        .filter((item) => item.source === 'upload')
        .forEach((item) => URL.revokeObjectURL(item.image));
    };
  }, [scenes]);

  const resetView = () => {
    setZoom(100);
    setOffset({ x: 0, y: 0 });
  };

  const selectScene = (id: string) => {
    setSelectedId(id);
    resetView();
  };

  const moveScene = (direction: -1 | 1) => {
    const next = (selectedIndex + direction + scenes.length) % scenes.length;
    selectScene(scenes[next].id);
  };

  const handleUpload = (files: FileList | null) => {
    const file = files?.[0];
    if (!file || !file.type.startsWith('image/')) return;
    const objectUrl = URL.createObjectURL(file);
    const id = `upload-${Date.now()}`;
    const uploadedScene: Scene = {
      id,
      title: file.name.replace(/\.[^/.]+$/, ''),
      subtitle: 'Local upload',
      image: objectUrl,
      map: '/data/episode-03.jpg',
      view: 'Uploaded view',
      step: '—',
      distance: 'Unknown',
      heading: 'Unknown',
      sceneType: 'Custom',
      instruction: 'Add spatial annotations to this locally loaded research image.',
      source: 'upload',
    };
    setScenes((current) => [...current, uploadedScene]);
    selectScene(id);
  };

  const addAnnotation = (event: React.MouseEvent<HTMLDivElement>) => {
    if (!annotationMode) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const point: Annotation = {
      id: `${scene.id}-${Date.now()}`,
      type: annotationType,
      x: clamp(((event.clientX - rect.left) / rect.width) * 100, 0, 100),
      y: clamp(((event.clientY - rect.top) / rect.height) * 100, 0, 100),
    };
    setAnnotations((current) => ({
      ...current,
      [scene.id]: [...(current[scene.id] ?? []), point],
    }));
  };

  const removeAnnotation = (id: string) => {
    setAnnotations((current) => ({
      ...current,
      [scene.id]: (current[scene.id] ?? []).filter((point) => point.id !== id),
    }));
  };

  const exportAnnotations = () => {
    const payload = {
      dataset: 'Lost on Campus sample',
      exportedAt: new Date().toISOString(),
      annotations,
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], {
      type: 'application/json',
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'locus-annotations.json';
    link.click();
    URL.revokeObjectURL(url);
  };

  const onPointerDown = (event: React.PointerEvent<HTMLDivElement>) => {
    if (annotationMode) return;
    event.currentTarget.setPointerCapture(event.pointerId);
    dragRef.current = {
      active: true,
      startX: event.clientX,
      startY: event.clientY,
      originX: offset.x,
      originY: offset.y,
    };
  };

  const onPointerMove = (event: React.PointerEvent<HTMLDivElement>) => {
    if (!dragRef.current.active || annotationMode) return;
    setOffset({
      x: dragRef.current.originX + event.clientX - dragRef.current.startX,
      y: dragRef.current.originY + event.clientY - dragRef.current.startY,
    });
  };

  const stopDragging = () => {
    dragRef.current.active = false;
  };

  return (
    <main className="min-h-screen bg-background text-foreground">
      <header className="sticky top-0 z-30 border-b border-white/10 bg-[#09120f]/95 px-4 py-3 text-white backdrop-blur-xl lg:px-7">
        <div className="mx-auto flex max-w-[1680px] items-center justify-between gap-4">
          <div className="flex min-w-0 items-center gap-3">
            <div className="grid size-9 shrink-0 place-items-center rounded-xl bg-[#c9ff8c] text-[#102016] shadow-[0_0_22px_rgba(201,255,140,.16)]">
              <Compass className="size-5" />
            </div>
            <div className="min-w-0">
              <p className="text-sm font-semibold tracking-[0.15em]">LOCUS</p>
              <p className="truncate text-[11px] text-white/48">Campus spatial episode viewer</p>
            </div>
          </div>

          <div className="hidden items-center gap-2 xl:flex">
            <span className="header-stat"><strong>15</strong> scenes</span>
            <span className="header-stat"><strong>50k+</strong> frames</span>
            <span className="header-stat"><strong>6k</strong> QA pairs</span>
          </div>

          <div className="flex items-center gap-2">
            <Badge className="hidden bg-white/10 text-white hover:bg-white/10 sm:inline-flex">Research prototype · HW1</Badge>
            <input
              ref={fileInputRef}
              className="hidden"
              type="file"
              accept="image/*"
              onChange={(event) => handleUpload(event.target.files)}
            />
            <Button
              variant="outline"
              size="sm"
              className="border-white/15 bg-white/5 text-white hover:bg-white/10 hover:text-white"
              onClick={() => fileInputRef.current?.click()}
            >
              <ImagePlus /> <span className="hidden sm:inline">Add image</span>
            </Button>
          </div>
        </div>
      </header>

      <section className="mx-auto grid max-w-[1680px] gap-4 p-3 sm:p-4 xl:grid-cols-[236px_minmax(0,1fr)_332px] xl:p-5">
        <aside className="surface order-2 overflow-hidden xl:order-1">
          <div className="flex items-center justify-between border-b border-border px-4 py-3.5">
            <div>
              <p className="eyebrow">Dataset browser</p>
              <h2 className="mt-1 text-base font-semibold">Navigation views</h2>
            </div>
            <Badge variant="outline" className="font-mono text-[10px]">{scenes.length}</Badge>
          </div>
          <div className="flex gap-2 overflow-x-auto p-3 xl:max-h-[calc(100vh-132px)] xl:flex-col xl:overflow-y-auto">
            {scenes.map((item, index) => {
              const count = annotations[item.id]?.length ?? 0;
              const active = item.id === scene.id;
              return (
                <button
                  key={item.id}
                  onClick={() => selectScene(item.id)}
                  className={`group min-w-[180px] rounded-xl border p-2 text-left transition xl:min-w-0 ${
                    active
                      ? 'border-[#6c9865] bg-[#e8f2de] shadow-[0_8px_20px_rgba(43,84,55,.08)]'
                      : 'border-transparent hover:border-border hover:bg-muted/70'
                  }`}
                >
                  <div className="relative overflow-hidden rounded-lg bg-[#101714]">
                    <img
                      src={item.image}
                      alt=""
                      className="aspect-[4/3] w-full object-cover transition duration-300 group-hover:scale-[1.03]"
                    />
                    <span className="absolute left-2 top-2 rounded-md bg-black/55 px-1.5 py-0.5 font-mono text-[9px] text-white backdrop-blur">
                      {String(index + 1).padStart(2, '0')}
                    </span>
                    {count > 0 && (
                      <span className="absolute bottom-2 right-2 flex items-center gap-1 rounded-full bg-[#c9ff8c] px-1.5 py-0.5 text-[9px] font-bold text-[#173020]">
                        <CircleDot className="size-2.5" /> {count}
                      </span>
                    )}
                  </div>
                  <div className="mt-2 flex items-start justify-between gap-2">
                    <div className="min-w-0">
                      <p className="truncate text-xs font-semibold">{item.title}</p>
                      <p className="truncate text-[10px] text-muted-foreground">{item.subtitle}</p>
                    </div>
                    <span className="font-mono text-[9px] text-muted-foreground">{item.step}</span>
                  </div>
                </button>
              );
            })}
          </div>
        </aside>

        <section className="surface order-1 min-w-0 overflow-hidden xl:order-2">
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border px-4 py-3">
            <div className="min-w-0">
              <div className="flex items-center gap-2">
                <p className="eyebrow">Egocentric observation</p>
                <span className="text-muted-foreground/35">/</span>
                <p className="font-mono text-[10px] text-muted-foreground">STEP {scene.step}</p>
              </div>
              <h1 className="mt-0.5 truncate text-lg font-semibold tracking-tight sm:text-xl">{scene.title}</h1>
            </div>
            <div className="flex items-center gap-1.5">
              <Button variant="outline" size="icon-sm" aria-label="Previous view" onClick={() => moveScene(-1)}>
                <ChevronLeft />
              </Button>
              <Button variant="outline" size="icon-sm" aria-label="Next view" onClick={() => moveScene(1)}>
                <ChevronRight />
              </Button>
              <div className="mx-1 h-5 w-px bg-border" />
              <Button
                size="sm"
                aria-pressed={annotationMode}
                className={annotationMode ? 'bg-[#c9ff8c] text-[#173020] hover:bg-[#b7ed7d]' : 'bg-[#244c37] hover:bg-[#183628]'}
                onClick={() => setAnnotationMode((current) => !current)}
              >
                {annotationMode ? <MousePointer2 /> : <ScanSearch />}
                {annotationMode ? 'Click to label' : 'Annotate'}
              </Button>
            </div>
          </div>

          {annotationMode && (
            <div className="flex flex-wrap items-center gap-2 border-b border-[#bdd4ae] bg-[#edf5e6] px-4 py-2.5">
              <span className="mr-1 text-[10px] font-bold uppercase tracking-wider text-[#46633e]">Label as</span>
              {annotationTypes.map((type) => {
                const Icon = type.icon;
                const active = annotationType === type.id;
                return (
                  <button
                    key={type.id}
                    onClick={() => setAnnotationType(type.id)}
                    className={`flex items-center gap-1.5 rounded-lg border px-2.5 py-1 text-xs font-medium transition ${
                      active ? 'border-[#52734d] bg-white text-[#213d2c] shadow-sm' : 'border-transparent text-[#53704e] hover:bg-white/60'
                    }`}
                  >
                    <Icon className="size-3.5" style={{ color: type.color === '#b8f36a' ? '#4a713b' : type.color }} />
                    {type.label}
                  </button>
                );
              })}
              <span className="ml-auto hidden text-[10px] text-[#62755d] sm:block">Click anywhere on the image to place a point</span>
            </div>
          )}

          <div
            className={`viewer-stage relative flex min-h-[520px] items-center justify-center overflow-hidden bg-[#0b110f] p-5 sm:min-h-[620px] ${annotationMode ? 'cursor-crosshair' : 'cursor-grab active:cursor-grabbing'}`}
            onPointerDown={onPointerDown}
            onPointerMove={onPointerMove}
            onPointerUp={stopDragging}
            onPointerCancel={stopDragging}
            onWheel={(event) => {
              event.preventDefault();
              setZoom((current) => clamp(current + (event.deltaY > 0 ? -10 : 10), 50, 250));
            }}
          >
            <div
              className="relative inline-block touch-none transition-transform duration-75"
              style={{ transform: `translate(${offset.x}px, ${offset.y}px) scale(${zoom / 100})` }}
              onClick={addAnnotation}
            >
              <img
                src={scene.image}
                alt={`${scene.view}: ${scene.title}, a campus navigation observation`}
                draggable={false}
                className="block h-[min(66vh,680px)] max-w-[min(100%,900px)] select-none object-contain shadow-[0_28px_80px_rgba(0,0,0,.42)]"
              />
              {sceneAnnotations.map((point, index) => {
                const meta = annotationTypes.find((type) => type.id === point.type)!;
                return (
                  <span
                    key={point.id}
                    className="annotation-pin"
                    style={{ left: `${point.x}%`, top: `${point.y}%`, '--pin-color': meta.color } as React.CSSProperties}
                    title={`${meta.label} ${index + 1}`}
                  >
                    {index + 1}
                  </span>
                );
              })}
            </div>

            <div className="pointer-events-none absolute left-4 top-4 flex items-center gap-2 rounded-full border border-white/10 bg-black/45 px-3 py-1.5 text-[10px] text-white/70 backdrop-blur-md">
              <span className="size-1.5 rounded-full bg-[#c9ff8c] shadow-[0_0_9px_#c9ff8c]" />
              {scene.view} · 120° FoV
            </div>

            <div className="absolute bottom-4 left-1/2 flex w-[min(520px,calc(100%-32px))] -translate-x-1/2 items-center gap-2 rounded-xl border border-white/10 bg-[#111a17]/88 p-2 text-white shadow-2xl backdrop-blur-xl">
              <Button variant="ghost" size="icon-sm" className="text-white hover:bg-white/10 hover:text-white" aria-label="Zoom out" onClick={() => setZoom((value) => clamp(value - 10, 50, 250))}>
                <Minus />
              </Button>
              <input
                type="range"
                aria-label="Zoom"
                min={50}
                max={250}
                step={10}
                value={zoom}
                onChange={(event) => setZoom(Number(event.target.value))}
                className="zoom-range min-w-0 flex-1"
              />
              <Button variant="ghost" size="icon-sm" className="text-white hover:bg-white/10 hover:text-white" aria-label="Zoom in" onClick={() => setZoom((value) => clamp(value + 10, 50, 250))}>
                <Plus />
              </Button>
              <span className="w-11 text-center font-mono text-[10px] text-white/60">{zoom}%</span>
              <Button variant="ghost" size="icon-sm" className="text-white hover:bg-white/10 hover:text-white" aria-label="Reset view" onClick={resetView}>
                <RotateCcw />
              </Button>
            </div>
          </div>
        </section>

        <aside className="surface order-3 overflow-hidden">
          <Tabs defaultValue="context" className="gap-0">
            <div className="flex items-center justify-between border-b border-border px-4 py-3">
              <div>
                <p className="eyebrow">Spatial workspace</p>
                <h2 className="mt-1 text-base font-semibold">Episode details</h2>
              </div>
              <TabsList>
                <TabsTrigger value="context">Context</TabsTrigger>
                <TabsTrigger value="labels">Labels</TabsTrigger>
              </TabsList>
            </div>

            <TabsContent value="context" className="space-y-4 p-4">
              <div className="group relative overflow-hidden rounded-xl border border-border bg-[#101714]">
                <img src={scene.map} alt="Bird's-eye route map for the selected navigation episode" className="aspect-[4/3] w-full object-cover transition duration-300 group-hover:scale-[1.02]" />
                <div className="absolute inset-x-2 bottom-2 flex items-center justify-between rounded-lg bg-black/55 px-2.5 py-1.5 text-[10px] text-white backdrop-blur">
                  <span className="flex items-center gap-1.5"><MapPinned className="size-3" /> BEV route</span>
                  <span className="font-mono">START → GOAL</span>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-2">
                <div className="metric"><span>Goal distance</span><strong>{scene.distance}</strong></div>
                <div className="metric"><span>Heading</span><strong>{scene.heading}</strong></div>
                <div className="metric"><span>Scene type</span><strong>{scene.sceneType}</strong></div>
                <div className="metric"><span>Annotations</span><strong>{sceneAnnotations.length}</strong></div>
              </div>

              <div className="rounded-xl bg-[#edf3e6] p-4">
                <div className="flex items-center gap-2 text-[#244c37]">
                  <Route className="size-4" />
                  <span className="text-[10px] font-bold uppercase tracking-wider">Route instruction</span>
                </div>
                <p className="mt-2 text-sm leading-6 text-[#2f4035]">{scene.instruction}</p>
              </div>

              <div className="rounded-xl border border-border p-4">
                <div className="flex items-center gap-2"><Info className="size-4 text-muted-foreground" /><span className="text-xs font-semibold">Research context</span></div>
                <p className="mt-2 text-xs leading-5 text-muted-foreground">
                  Lost on Campus evaluates action grounding, spatial foresight, metric awareness, goal-directed planning, ego-allocentric localization, and spatio-temporal consistency in outdoor 3DGS scenes.
                </p>
              </div>
            </TabsContent>

            <TabsContent value="labels" className="p-4">
              <div className="flex items-center justify-between gap-2">
                <div>
                  <p className="text-sm font-semibold">Spatial annotations</p>
                  <p className="text-[10px] text-muted-foreground">Saved locally in this browser</p>
                </div>
                <Button variant="outline" size="sm" onClick={exportAnnotations} disabled={Object.keys(annotations).length === 0}>
                  <Download /> Export
                </Button>
              </div>

              {sceneAnnotations.length === 0 ? (
                <div className="mt-6 grid place-items-center rounded-xl border border-dashed border-border bg-muted/35 px-5 py-10 text-center">
                  <div className="grid size-10 place-items-center rounded-full bg-card shadow-sm"><Focus className="size-4 text-muted-foreground" /></div>
                  <p className="mt-3 text-sm font-medium">No labels on this view</p>
                  <p className="mt-1 max-w-[220px] text-xs leading-5 text-muted-foreground">Turn on Annotate, choose a category, then click a spatial feature in the image.</p>
                  <Button className="mt-4 bg-[#244c37] hover:bg-[#183628]" size="sm" onClick={() => setAnnotationMode(true)}>
                    <ScanSearch /> Start labeling
                  </Button>
                </div>
              ) : (
                <div className="mt-4 space-y-2">
                  {sceneAnnotations.map((point, index) => {
                    const meta = annotationTypes.find((type) => type.id === point.type)!;
                    const Icon = meta.icon;
                    return (
                      <div key={point.id} className="flex items-center gap-3 rounded-xl border border-border bg-muted/30 p-3">
                        <span className="grid size-8 shrink-0 place-items-center rounded-lg" style={{ backgroundColor: `${meta.color}40`, color: '#294934' }}><Icon className="size-4" /></span>
                        <div className="min-w-0 flex-1">
                          <p className="text-xs font-semibold">{meta.label} {index + 1}</p>
                          <p className="font-mono text-[9px] text-muted-foreground">x {point.x.toFixed(1)} · y {point.y.toFixed(1)}</p>
                        </div>
                        <Button variant="ghost" size="icon-sm" aria-label={`Delete ${meta.label} annotation`} onClick={() => removeAnnotation(point.id)}>
                          <Trash2 />
                        </Button>
                      </div>
                    );
                  })}
                </div>
              )}
            </TabsContent>
          </Tabs>
        </aside>
      </section>

      <footer className="mx-auto flex max-w-[1680px] flex-wrap items-center justify-between gap-3 px-5 pb-6 pt-1 text-[10px] text-muted-foreground">
        <span className="flex items-center gap-1.5"><FileImage className="size-3" /> Sample views derived from the Lost on Campus research paper.</span>
        <span>Interactive image viewer · zoom · pan · local upload · point annotation · JSON export</span>
      </footer>
    </main>
  );
}

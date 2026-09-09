import {
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type CSSProperties,
  type PointerEvent as ReactPointerEvent,
} from "react";

import type { Annotation, BoundingBox, Label, ProjectImage, UUID } from "../types";
import {
  annotationTargetsAtPoint,
  anchoredScrollDelta,
  canvasPointFromClient,
  clamp,
  movedBox,
  normalizedBox,
  resizedBox,
  zoomAnchorFromClient,
  type MoveGesture,
  type Point,
  type ResizeGesture,
  type ResizeHandle,
} from "./annotationGeometry";
import { layoutCanvasLabels, type CanvasLabelMode } from "./canvasLabels";

type Gesture =
  | { type: "draw"; start: Point; current: Point }
  | (MoveGesture & { cycleCandidates: UUID[] })
  | ResizeGesture
  | { type: "select"; start: Point; current: Point; cycleCandidates: UUID[] };

interface AnnotationCanvasProps {
  image: ProjectImage;
  imageUrl: string;
  labels: Label[];
  selectedLabelId: UUID | null;
  selectedAnnotationId: UUID | null;
  interactionMode?: "select" | "draw";
  zoom?: number;
  labelMode?: CanvasLabelMode;
  disabled?: boolean;
  onZoomChange?: (zoom: number) => void;
  onSelectAnnotation: (annotationId: UUID | null) => void;
  onCreate: (box: BoundingBox) => Promise<void>;
  onUpdate: (annotationId: UUID, box: BoundingBox, action: "Move box" | "Resize box") => Promise<void>;
}

function handlePoint(box: BoundingBox, handle: ResizeHandle): Point {
  return {
    x: handle.includes("w") ? box.x : box.x + box.width,
    y: handle.includes("n") ? box.y : box.y + box.height,
  };
}

export function AnnotationCanvas({
  image,
  imageUrl,
  labels,
  selectedLabelId,
  selectedAnnotationId,
  interactionMode = "draw",
  zoom = 1,
  labelMode = "hover",
  disabled = false,
  onZoomChange,
  onSelectAnnotation,
  onCreate,
  onUpdate,
}: AnnotationCanvasProps) {
  const [gesture, setGestureState] = useState<Gesture | null>(null);
  const [viewportSize, setViewportSize] = useState({ width: 0, height: 0 });
  const [spacePressed, setSpacePressed] = useState(false);
  const [panning, setPanning] = useState(false);
  const [hoveredAnnotationId, setHoveredAnnotationId] = useState<UUID | null>(null);
  const gestureRef = useRef<Gesture | null>(null);
  const viewportRef = useRef<HTMLDivElement>(null);
  const stageRef = useRef<HTMLDivElement>(null);
  const zoomRef = useRef(zoom);
  const onZoomChangeRef = useRef(onZoomChange);
  const pendingZoomAnchorRef = useRef<ReturnType<typeof zoomAnchorFromClient> | null>(null);
  const panRef = useRef<{
    pointerId: number;
    clientX: number;
    clientY: number;
    scrollLeft: number;
    scrollTop: number;
  } | null>(null);
  const unit = Math.max(image.width, image.height) / 900;
  const handleRadius = unit * 6;
  const strokeWidth = unit * 2;
  const targetsAtPoint = (point: Point) =>
    annotationTargetsAtPoint(
      image.annotations.flatMap((annotation) => {
        const box = annotation.final_box ?? annotation.original_ai_box;
        return box && labels.some((label) => label.id === annotation.label_id) ? [{ id: annotation.id, box }] : [];
      }),
      point,
      handleRadius,
    );

  zoomRef.current = zoom;
  onZoomChangeRef.current = onZoomChange;

  useLayoutEffect(() => {
    const viewport = viewportRef.current;
    if (!viewport) return;
    const updateSize = () => setViewportSize({ width: viewport.clientWidth, height: viewport.clientHeight });
    updateSize();
    if (typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(updateSize);
    observer.observe(viewport);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const updateSpace = (event: KeyboardEvent, pressed: boolean) => {
      const target = event.target;
      if (
        target instanceof HTMLInputElement ||
        target instanceof HTMLTextAreaElement ||
        target instanceof HTMLSelectElement ||
        (target instanceof HTMLElement && target.isContentEditable)
      ) {
        return;
      }
      if (event.code === "Space") {
        if (pressed) event.preventDefault();
        setSpacePressed(pressed);
      }
    };
    const keyDown = (event: KeyboardEvent) => updateSpace(event, true);
    const keyUp = (event: KeyboardEvent) => updateSpace(event, false);
    window.addEventListener("keydown", keyDown);
    window.addEventListener("keyup", keyUp);
    return () => {
      window.removeEventListener("keydown", keyDown);
      window.removeEventListener("keyup", keyUp);
    };
  }, []);

  useEffect(() => setHoveredAnnotationId(null), [image.id]);

  useEffect(() => {
    const viewport = viewportRef.current;
    if (!viewport) return;
    const zoomCanvas = (event: WheelEvent) => {
      const changeZoom = onZoomChangeRef.current;
      if (!event.ctrlKey || !changeZoom) return;
      event.preventDefault();
      const currentZoom = zoomRef.current;
      const nextZoom = clamp(currentZoom + (event.deltaY < 0 ? 0.25 : -0.25), 0.5, 4);
      if (nextZoom === currentZoom) return;
      const stage = stageRef.current;
      pendingZoomAnchorRef.current = stage
        ? zoomAnchorFromClient(stage.getBoundingClientRect(), event.clientX, event.clientY)
        : null;
      changeZoom(nextZoom);
    };
    viewport.addEventListener("wheel", zoomCanvas, { passive: false });
    return () => viewport.removeEventListener("wheel", zoomCanvas);
  }, []);

  const setGesture = (next: Gesture | null) => {
    gestureRef.current = next;
    setGestureState(next);
  };

  const pointFromEvent = (event: ReactPointerEvent<SVGSVGElement>): Point => {
    const bounds = event.currentTarget.getBoundingClientRect();
    return canvasPointFromClient(bounds, image, event.clientX, event.clientY);
  };

  const capturePointer = (event: ReactPointerEvent<SVGElement>, nextGesture: Gesture) => {
    event.stopPropagation();
    const svg = event.currentTarget.ownerSVGElement ?? (event.currentTarget as SVGSVGElement);
    svg.setPointerCapture(event.pointerId);
    setGesture(nextGesture);
  };

  const startCanvasGesture = (event: ReactPointerEvent<SVGSVGElement>) => {
    if (disabled || event.button !== 0 || spacePressed) return;
    const point = pointFromEvent(event);
    if (interactionMode === "select") {
      const candidates = targetsAtPoint(point);
      const annotationId = candidates[0];
      if (!annotationId) {
        onSelectAnnotation(null);
        return;
      }
      const annotation = image.annotations.find((item) => item.id === annotationId);
      const box = annotation?.final_box ?? annotation?.original_ai_box;
      const cycleCandidates = annotationId === selectedAnnotationId ? candidates.slice(1) : [];
      onSelectAnnotation(annotationId);
      event.currentTarget.setPointerCapture(event.pointerId);
      if (!annotation || !box || annotation.verification_state === "rejected") {
        setGesture({ type: "select", start: point, current: point, cycleCandidates });
        return;
      }
      setGesture({
        type: "move",
        annotationId,
        start: point,
        current: point,
        original: box,
        cycleCandidates,
      });
      return;
    }
    if (!selectedLabelId) return;
    onSelectAnnotation(null);
    event.currentTarget.setPointerCapture(event.pointerId);
    setGesture({ type: "draw", start: point, current: point });
  };

  const movePointer = (event: ReactPointerEvent<SVGSVGElement>) => {
    const active = gestureRef.current;
    if (!active) return;
    setGesture({ ...active, current: pointFromEvent(event) });
  };

  const finishPointer = () => {
    const active = gestureRef.current;
    if (!active) return;
    setGesture(null);
    if (active.type === "select") {
      if (active.current.x === active.start.x && active.current.y === active.start.y && active.cycleCandidates[0]) {
        onSelectAnnotation(active.cycleCandidates[0]);
      }
      return;
    }
    if (active.type === "draw") {
      const box = normalizedBox(active.start, active.current);
      if (box.width >= 4 && box.height >= 4) void onCreate(box);
      return;
    }
    const box = active.type === "move" ? movedBox(active, image) : resizedBox(active, image);
    if (
      box.x === active.original.x &&
      box.y === active.original.y &&
      box.width === active.original.width &&
      box.height === active.original.height
    ) {
      if (active.type === "move" && active.cycleCandidates[0]) onSelectAnnotation(active.cycleCandidates[0]);
      return;
    }
    void onUpdate(active.annotationId, box, active.type === "move" ? "Move box" : "Resize box");
  };

  const previewBox = (annotation: Annotation): BoundingBox | null => {
    const active = gesture;
    if (!active || active.type === "draw" || active.type === "select" || active.annotationId !== annotation.id) {
      return annotation.final_box ?? annotation.original_ai_box;
    }
    return active.type === "move" ? movedBox(active, image) : resizedBox(active, image);
  };

  const imageRatio = image.width / image.height;
  const availableWidth = viewportSize.width || image.width;
  const availableHeight = viewportSize.height || image.height;
  const fitScale = Math.min(availableWidth / image.width, availableHeight / image.height);
  const displayWidth = Math.max(1, image.width * fitScale * zoom);
  const displayHeight = Math.max(1, image.height * fitScale * zoom);
  const contentWidth = Math.max(availableWidth, displayWidth);
  const contentHeight = Math.max(availableHeight, displayHeight);
  const stageStyle = {
    "--image-ratio": imageRatio,
    position: "absolute",
    width: `${displayWidth}px`,
    height: `${displayHeight}px`,
    left: `${(contentWidth - displayWidth) / 2}px`,
    top: `${(contentHeight - displayHeight) / 2}px`,
  } as CSSProperties;

  useLayoutEffect(() => {
    const viewport = viewportRef.current;
    if (!viewport) return;
    const anchor = pendingZoomAnchorRef.current;
    const stage = stageRef.current;
    if (anchor && stage) {
      const delta = anchoredScrollDelta(stage.getBoundingClientRect(), anchor);
      viewport.scrollLeft += delta.x;
      viewport.scrollTop += delta.y;
      pendingZoomAnchorRef.current = null;
      return;
    }
    viewport.scrollLeft = Math.max(0, (viewport.scrollWidth - viewport.clientWidth) / 2);
    viewport.scrollTop = Math.max(0, (viewport.scrollHeight - viewport.clientHeight) / 2);
  }, [displayHeight, displayWidth, image.id]);

  const startPan = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (event.button !== 1 && !(event.button === 0 && spacePressed)) return;
    const viewport = viewportRef.current;
    if (!viewport) return;
    event.preventDefault();
    viewport.setPointerCapture(event.pointerId);
    panRef.current = {
      pointerId: event.pointerId,
      clientX: event.clientX,
      clientY: event.clientY,
      scrollLeft: viewport.scrollLeft,
      scrollTop: viewport.scrollTop,
    };
    setPanning(true);
  };

  const pan = (event: ReactPointerEvent<HTMLDivElement>) => {
    const active = panRef.current;
    const viewport = viewportRef.current;
    if (!active || !viewport || active.pointerId !== event.pointerId) return;
    viewport.scrollLeft = active.scrollLeft - (event.clientX - active.clientX);
    viewport.scrollTop = active.scrollTop - (event.clientY - active.clientY);
  };

  const finishPan = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (panRef.current?.pointerId !== event.pointerId) return;
    panRef.current = null;
    setPanning(false);
  };
  const canvasHelp =
    interactionMode === "draw" ? "Drag empty space to draw a box." : "Select, move, or resize an existing box.";
  const labelLayouts = new Map(
    layoutCanvasLabels(
      image.annotations.flatMap((annotation) => {
        const box = previewBox(annotation);
        const label = labels.find((candidate) => candidate.id === annotation.label_id);
        if (!box || !label) return [];
        return [
          {
            id: annotation.id,
            box,
            classText: label.name || "Unlabeled",
            stateText:
              annotation.verification_state === "unreviewed"
                ? "AI proposal"
                : annotation.verification_state === "human_added"
                  ? "Human added"
                  : annotation.verification_state[0].toUpperCase() + annotation.verification_state.slice(1),
            priority:
              annotation.id === selectedAnnotationId
                ? ("selected" as const)
                : annotation.id === hoveredAnnotationId
                  ? ("hovered" as const)
                  : ("normal" as const),
          },
        ];
      }),
      image,
      unit,
      labelMode,
    ).map((layout) => [layout.id, layout]),
  );
  const renderAnnotations = [...image.annotations].sort((left, right) => {
    const leftPriority = left.id === hoveredAnnotationId ? 2 : left.id === selectedAnnotationId ? 1 : 0;
    const rightPriority = right.id === hoveredAnnotationId ? 2 : right.id === selectedAnnotationId ? 1 : 0;
    return leftPriority - rightPriority;
  });
  const selectedAnnotation = image.annotations.find((item) => item.id === selectedAnnotationId);
  const selectedBox = selectedAnnotation ? previewBox(selectedAnnotation) : null;
  const selectedLabel = labels.find((item) => item.id === selectedAnnotation?.label_id);

  return (
    <div className="canvas-shell">
      <div
        ref={viewportRef}
        className={`canvas-viewport ${spacePressed ? "can-pan" : ""} ${panning ? "is-panning" : ""}`}
        onPointerDown={startPan}
        onPointerMove={pan}
        onPointerUp={finishPan}
        onPointerCancel={finishPan}
      >
        <div className="canvas-scroll-content" style={{ width: `${contentWidth}px`, height: `${contentHeight}px` }}>
          <div ref={stageRef} className="annotation-stage" style={stageStyle}>
            <img src={imageUrl} alt={image.filename} className="workspace-image" draggable={false} />
            <svg
              className={`annotation-overlay ${interactionMode === "draw" && selectedLabelId && !disabled ? "is-drawing" : ""}`}
              viewBox={`0 0 ${image.width} ${image.height}`}
              role="application"
              tabIndex={0}
              aria-label={`Annotation canvas for ${image.filename}. ${canvasHelp}`}
              onPointerDown={startCanvasGesture}
              onPointerMove={(event) => {
                movePointer(event);
                if (gestureRef.current) return;
                const point = pointFromEvent(event);
                const hovered = targetsAtPoint(point)[0];
                setHoveredAnnotationId(hovered ?? null);
              }}
              onPointerUp={finishPointer}
              onPointerCancel={() => setGesture(null)}
              onPointerLeave={() => {
                if (!gestureRef.current) setHoveredAnnotationId(null);
              }}
            >
              {renderAnnotations.map((annotation) => {
                const box = previewBox(annotation);
                const label = labels.find((candidate) => candidate.id === annotation.label_id);
                if (!box || !label) return null;
                const selected = annotation.id === selectedAnnotationId;
                const hovered = annotation.id === hoveredAnnotationId;
                const unresolved = annotation.verification_state === "unreviewed";
                const rejected = annotation.verification_state === "rejected";
                const adjusted = annotation.verification_state === "adjusted";
                const humanAdded = annotation.verification_state === "human_added";
                const visualColor = rejected
                  ? "var(--color-rejected)"
                  : unresolved
                    ? "var(--color-adjusted)"
                    : humanAdded
                      ? "var(--color-human)"
                      : label.color;
                const stateText =
                  annotation.verification_state === "unreviewed"
                    ? "Needs review"
                    : annotation.verification_state === "human_added"
                      ? "Human added"
                      : annotation.verification_state[0].toUpperCase() + annotation.verification_state.slice(1);
                const classText = label.name || "Unlabeled";
                const labelLayout = labelLayouts.get(annotation.id);
                return (
                  <g
                    key={annotation.id}
                    data-annotation-id={annotation.id}
                    className={`annotation-layer ${selected ? "is-selected" : ""} ${hovered ? "is-hovered" : ""}`}
                  >
                    {selected && adjusted && annotation.original_ai_box && (
                      <rect
                        x={annotation.original_ai_box.x}
                        y={annotation.original_ai_box.y}
                        width={annotation.original_ai_box.width}
                        height={annotation.original_ai_box.height}
                        fill="none"
                        stroke="var(--color-adjusted)"
                        strokeWidth={strokeWidth}
                        strokeDasharray={`${unit * 6} ${unit * 5}`}
                        opacity="0.65"
                        pointerEvents="none"
                      />
                    )}
                    <rect
                      x={box.x}
                      y={box.y}
                      width={box.width}
                      height={box.height}
                      fill={selected ? label.color : visualColor}
                      fillOpacity={selected ? 0.12 : 0.08}
                      stroke={selected ? label.color : visualColor}
                      strokeWidth={selected ? strokeWidth * 1.6 : hovered ? strokeWidth * 1.35 : strokeWidth}
                      strokeDasharray={
                        selected
                          ? `${unit * 7} ${unit * 4}`
                          : unresolved
                            ? `${unit * 9} ${unit * 5}`
                            : rejected
                              ? `${unit * 4} ${unit * 5}`
                              : undefined
                      }
                      opacity={rejected ? 0.72 : 1}
                      className={`annotation-box state-${annotation.verification_state}`}
                      pointerEvents="none"
                    />
                    {labelLayout && (
                      <g
                        className={`box-label ${labelLayout.detail ? "has-detail" : ""} ${selected ? "selected" : ""}`}
                        pointerEvents="none"
                      >
                        <rect
                          x={labelLayout.x}
                          y={labelLayout.y}
                          width={labelLayout.width}
                          height={labelLayout.height}
                          rx={5 * unit}
                          fill="var(--color-floating-solid)"
                          stroke={selected ? label.color : visualColor}
                          strokeWidth={Math.max(unit, strokeWidth * 0.55)}
                        />
                        <circle
                          cx={labelLayout.x + 11 * unit}
                          cy={labelLayout.y + (labelLayout.detail ? 14 : 14.5) * unit}
                          r={3.5 * unit}
                          fill={selected ? label.color : visualColor}
                        />
                        <text
                          x={labelLayout.x + 20 * unit}
                          y={labelLayout.y + 19 * unit}
                          fontSize={17 * unit}
                          fontWeight="700"
                          fill="var(--color-text-strong)"
                        >
                          {classText}
                        </text>
                        {labelLayout.detail && (
                          <text
                            x={labelLayout.x + 20 * unit}
                            y={labelLayout.y + 35 * unit}
                            fontSize={13 * unit}
                            fontWeight="600"
                            fill="var(--color-text-muted)"
                          >
                            {stateText}
                          </text>
                        )}
                      </g>
                    )}
                    {rejected && (
                      <g pointerEvents="none" stroke="var(--color-rejected)" strokeWidth={strokeWidth * 1.4}>
                        <line x1={box.x} y1={box.y} x2={box.x + box.width} y2={box.y + box.height} />
                        <line x1={box.x + box.width} y1={box.y} x2={box.x} y2={box.y + box.height} />
                      </g>
                    )}
                  </g>
                );
              })}
              {selectedAnnotation &&
                selectedBox &&
                selectedLabel &&
                selectedAnnotation.verification_state !== "rejected" &&
                (["nw", "ne", "se", "sw"] as ResizeHandle[]).map((handle) => {
                  const point = handlePoint(selectedBox, handle);
                  return (
                    <circle
                      key={handle}
                      cx={point.x}
                      cy={point.y}
                      r={handleRadius}
                      fill="var(--color-text-strong)"
                      stroke={selectedLabel.color}
                      strokeWidth={strokeWidth}
                      className={`resize-handle resize-${handle}`}
                      onPointerDown={(event) => {
                        if (disabled || event.button !== 0) return;
                        capturePointer(event, {
                          type: "resize",
                          annotationId: selectedAnnotation.id,
                          handle,
                          start: point,
                          current: point,
                          original: selectedBox,
                        });
                      }}
                    />
                  );
                })}
              {gesture?.type === "draw" &&
                (() => {
                  const box = normalizedBox(gesture.start, gesture.current);
                  const color = "var(--color-human)";
                  return (
                    <rect
                      x={box.x}
                      y={box.y}
                      width={box.width}
                      height={box.height}
                      fill={`${color}20`}
                      stroke={color}
                      strokeWidth={strokeWidth}
                      strokeDasharray={`${unit * 8} ${unit * 5}`}
                      pointerEvents="none"
                    />
                  );
                })()}
            </svg>
          </div>
        </div>
      </div>
    </div>
  );
}

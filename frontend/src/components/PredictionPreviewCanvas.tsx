import type { PredictionPreview } from "../types";

export function PredictionPreviewCanvas({
  preview,
  imageUrl,
  className = "",
}: {
  preview: PredictionPreview;
  imageUrl: string;
  className?: string;
}) {
  return (
    <div className={`prediction-canvas-wrap ${className}`.trim()}>
      <svg
        viewBox={`0 0 ${preview.width} ${preview.height}`}
        role="img"
        aria-label={`Trained model predictions on ${preview.filename}`}
      >
        <image href={imageUrl} width={preview.width} height={preview.height} />
        {preview.detections.map((detection, index) => (
          <g key={`${detection.class_id}-${index}`}>
            <rect
              className="trained-prediction-box"
              x={detection.box.x}
              y={detection.box.y}
              width={detection.box.width}
              height={detection.box.height}
              vectorEffect="non-scaling-stroke"
            />
            <text x={detection.box.x + 3} y={Math.max(11, detection.box.y + 11)}>
              {detection.class_name} {Math.round(detection.confidence * 100)}%
            </text>
          </g>
        ))}
      </svg>
      <div className="prediction-caption">
        <strong>{preview.filename}</strong>
        <span>
          {preview.detections.length} trained-model detection
          {preview.detections.length === 1 ? "" : "s"}
        </span>
        <em>{preview.source === "uploaded_image" ? "Held-out upload" : "Project image"}</em>
      </div>
    </div>
  );
}

import React from "react";

// Structure, not furniture — drawn as the substrate the rest sits on.
// Mirrors twin/generator.py's WALKABLE_LABELS.
const SUBSTRATE = new Set(["floor", "ceiling"]);
const PAD = 0.5; // metres of margin around the scanned extent

/** The robot's executed path, ending where it parked. */
function Robot({ trace, place, size }) {
  const points = trace.map(([x, y]) => place(x, y).join(",")).join(" ");
  const [endX, endY] = place(trace[trace.length - 1][0], trace[trace.length - 1][1]);
  return (
    <>
      <polyline className="trace" points={points} strokeWidth={size / 100} />
      <circle className="robot" cx={endX} cy={endY} r={size / 90} />
    </>
  );
}

/**
 * Plan view of the scan, in metres, straight from /segment's `bbox3d` values
 * ([xmin, ymin, zmin, xmax, ymax, zmax], Z-up map frame — so x/y is the top-down
 * footprint) and /deploy's `pose_trace`.
 *
 * Screen y grows downward and the map's doesn't, so every point is projected once
 * through `place()` rather than flipping the SVG (which would mirror the labels too).
 */
export default function Floorplan({ objects, poseTrace }) {
  if (!objects.length) {
    return (
      <div className="plan empty-plan">
        <p>No plan view yet.</p>
        <p className="hint">Run stage 03 — segmentation is what puts objects on the floor.</p>
      </div>
    );
  }

  const xs = objects.flatMap((o) => [o.bbox3d[0], o.bbox3d[3]]);
  const ys = objects.flatMap((o) => [o.bbox3d[1], o.bbox3d[4]]);
  const minX = Math.min(...xs) - PAD;
  const maxX = Math.max(...xs) + PAD;
  const minY = Math.min(...ys) - PAD;
  const maxY = Math.max(...ys) + PAD;
  const width = maxX - minX;
  const height = maxY - minY;

  const place = (x, y) => [x - minX, maxY - y];
  const rect = (bbox) => {
    const [x, y] = place(bbox[0], bbox[4]);
    return { x, y, width: bbox[3] - bbox[0], height: bbox[4] - bbox[1] };
  };

  const gridlines = [];
  for (let x = Math.ceil(minX); x < maxX; x += 1) gridlines.push(["v", x - minX]);
  for (let y = Math.ceil(minY); y < maxY; y += 1) gridlines.push(["h", maxY - y]);

  const strokes = Math.max(width, height) / 400; // hairlines stay hairlines at any scale

  return (
    <svg className="plan" viewBox={`0 0 ${width} ${height}`} role="img"
         aria-label={`Plan view of ${objects.length} scanned objects`}>
      <g strokeWidth={strokes}>
        {gridlines.map(([axis, at], i) => (
          <line
            key={i}
            className="gridline"
            x1={axis === "v" ? at : 0}
            x2={axis === "v" ? at : width}
            y1={axis === "v" ? 0 : at}
            y2={axis === "v" ? height : at}
          />
        ))}
      </g>

      {objects.map((obj) => (
        <g key={obj.id} className={SUBSTRATE.has(obj.label) ? "substrate" : "prop"}>
          <rect {...rect(obj.bbox3d)} strokeWidth={strokes * 2} />
          {!SUBSTRATE.has(obj.label) && (
            <text
              x={rect(obj.bbox3d).x}
              y={rect(obj.bbox3d).y - strokes * 6}
              fontSize={Math.max(width, height) / 45}
            >
              {obj.label}
            </text>
          )}
        </g>
      ))}

      {poseTrace.length > 0 && <Robot trace={poseTrace} place={place} size={Math.max(width, height)} />}
    </svg>
  );
}

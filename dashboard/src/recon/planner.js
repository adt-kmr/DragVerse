/* A real (if small-scale) path planner for the hero's "Train" stage: given a start, a
   goal, and the room's furniture, it decides whether the direct line is clear and, if
   not, routes around whatever's in the way.

   This is a visibility-graph shortest path over axis-aligned obstacles — the same idea
   as the backend's navmesh-inflate-then-search approach in policy/evaluate.py, just
   continuous instead of grid-based, since the hero only ever has a handful of props to
   reason about. Obstacles are inflated by a clearance radius; the graph's nodes are the
   start, the goal, and every obstacle corner not swallowed by a neighbouring obstacle's
   own inflation; an edge exists between two nodes only if the straight line between them
   doesn't cut through any inflated obstacle. Dijkstra over that graph is the shortest
   collision-free route. */

function inflate(prop, pad) {
  return {
    prop,
    x0: prop.x - pad,
    y0: prop.y - pad,
    x1: prop.x + prop.w + pad,
    y1: prop.y + prop.d + pad,
    cx: prop.x + prop.w / 2,
    cy: prop.y + prop.d / 2,
  };
}

function corners(box) {
  return [
    [box.x0, box.y0],
    [box.x1, box.y0],
    [box.x1, box.y1],
    [box.x0, box.y1],
  ];
}

function pointInBox(p, box, eps = 1e-6) {
  return p[0] > box.x0 + eps && p[0] < box.x1 - eps && p[1] > box.y0 + eps && p[1] < box.y1 - eps;
}

function cross(a, b, c) {
  return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
}

/** True if open segments p1-p2 and p3-p4 cross. Collinear/touching segments (a route
    grazing an obstacle's edge, exactly the tangent line a real planner would take) count
    as clear, not blocked. */
function segmentsCross(p1, p2, p3, p4) {
  const d1 = cross(p3, p4, p1);
  const d2 = cross(p3, p4, p2);
  const d3 = cross(p1, p2, p3);
  const d4 = cross(p1, p2, p4);
  return (d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0) ? ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0)) : false;
}

/** True if segment a-b passes through box's interior — either endpoint lands inside it,
    or the segment crosses one of its four edges. */
function segmentHitsBox(a, b, box) {
  if (pointInBox(a, box) || pointInBox(b, box)) return true;
  const [c0, c1, c2, c3] = corners(box);
  return (
    segmentsCross(a, b, c0, c1) ||
    segmentsCross(a, b, c1, c2) ||
    segmentsCross(a, b, c2, c3) ||
    segmentsCross(a, b, c3, c0)
  );
}

function dist(a, b) {
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
}

/** Small O(n^2) Dijkstra — the graph never has more than a couple dozen nodes for a
    scene with a handful of props, so there's nothing to optimize here. */
function shortestPath(nodes, adj, startIdx, goalIdx) {
  const d = new Array(nodes.length).fill(Infinity);
  const prev = new Array(nodes.length).fill(-1);
  const seen = new Array(nodes.length).fill(false);
  d[startIdx] = 0;
  for (let iter = 0; iter < nodes.length; iter++) {
    let u = -1;
    let best = Infinity;
    for (let i = 0; i < nodes.length; i++) {
      if (!seen[i] && d[i] < best) {
        best = d[i];
        u = i;
      }
    }
    if (u === -1 || u === goalIdx) break;
    seen[u] = true;
    for (const [v, w] of adj[u]) {
      const nd = d[u] + w;
      if (nd < d[v]) {
        d[v] = nd;
        prev[v] = u;
      }
    }
  }
  if (d[goalIdx] === Infinity) return null;
  const path = [];
  for (let cur = goalIdx; cur !== -1; cur = prev[cur]) path.unshift(cur);
  return path.map((i) => nodes[i]);
}

/** planRoute(start, goal, props) -> { route, direct, blocked }
    - direct: the straight [start, goal] line the robot would take with no obstacles.
    - blocked: the props whose inflated footprint the direct line actually crosses,
      nearest-first, each with the straight-line distance from start.
    - route: the shortest collision-free polyline from start to goal. Equals `direct`
      when nothing is in the way. */
export function planRoute(start, goal, props, { padding = 0.35, exclude = ["robot"] } = {}) {
  const obstacles = props.filter((p) => !exclude.includes(p.label)).map((p) => inflate(p, padding));

  const blocked = obstacles
    .filter((box) => segmentHitsBox(start, goal, box))
    .map((box) => ({ prop: box.prop, dist: dist(start, [box.cx, box.cy]) }))
    .sort((a, b) => a.dist - b.dist);

  if (blocked.length === 0) {
    return { route: [start, goal], direct: [start, goal], blocked: [] };
  }

  const nodes = [start];
  for (const box of obstacles) {
    for (const c of corners(box)) {
      // A corner buried inside a neighbouring obstacle's own clearance isn't a usable
      // waypoint — any edge leaving it would immediately be blocked by that neighbour.
      if (obstacles.some((other) => other !== box && pointInBox(c, other))) continue;
      nodes.push(c);
    }
  }
  nodes.push(goal);
  const goalIdx = nodes.length - 1;

  const adj = nodes.map(() => []);
  for (let i = 0; i < nodes.length; i++) {
    for (let j = i + 1; j < nodes.length; j++) {
      const clear = !obstacles.some((box) => segmentHitsBox(nodes[i], nodes[j], box));
      if (clear) {
        const w = dist(nodes[i], nodes[j]);
        adj[i].push([j, w]);
        adj[j].push([i, w]);
      }
    }
  }

  const route = shortestPath(nodes, adj, 0, goalIdx) || [start, goal];
  return { route, direct: [start, goal], blocked };
}

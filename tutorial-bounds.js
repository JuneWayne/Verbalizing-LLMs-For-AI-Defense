import { Box3, Vector3 } from 'three';

// Project the visible solid into page pixels instead of highlighting its entire canvas.
export function tutorialBounds(objects, camera, canvas) {
  const rect = canvas.getBoundingClientRect();
  const points = [];
  for (const object of objects) {
    if (!object.visible) continue;
    object.updateWorldMatrix(true, true);
    const box = new Box3().setFromObject(object);
    if (box.isEmpty()) continue;
    for (const x of [box.min.x, box.max.x]) for (const y of [box.min.y, box.max.y]) for (const z of [box.min.z, box.max.z]) {
      const point = new Vector3(x, y, z).project(camera);
      points.push({x: rect.left + (point.x + 1) * rect.width / 2, y: rect.top + (1 - point.y) * rect.height / 2});
    }
  }
  if (!points.length) return null;
  const left = Math.min(...points.map(p => p.x)), right = Math.max(...points.map(p => p.x));
  const top = Math.min(...points.map(p => p.y)), bottom = Math.max(...points.map(p => p.y));
  return {left, top, right, bottom, width: right - left, height: bottom - top};
}

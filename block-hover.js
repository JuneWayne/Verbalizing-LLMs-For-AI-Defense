import * as THREE from 'three';
import { theme } from './theme.js';

// This frame has front, back, and side edges in the same 3D space as the block.
export function createBlockHover(parent, size, fillOpacity = .055) {
  const group = new THREE.Group();
  const geometry = new THREE.BoxGeometry(1, 1, 1);
  const edges = new THREE.LineSegments(new THREE.EdgesGeometry(geometry), new THREE.LineBasicMaterial({color: theme.accent, transparent: true, opacity: 0}));
  const fill = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({color: theme.accent, transparent: true, opacity: 0, depthWrite: false}));
  group.add(edges, fill);
  group.scale.copy(size).addScalar(.04);
  parent.add(group);
  let amount = 0, target = 0;
  function set(active, center, dimensions) {
    target = active ? 1 : 0;
    if (center) group.position.copy(center);
    if (dimensions) group.scale.copy(dimensions).addScalar(.04);
  }
  function animate(seconds, reduced) {
    amount += (target - amount) * (reduced ? 1 : 1 - Math.exp(-seconds * 24));
    group.visible = amount > .005;
    edges.material.opacity = amount;
    fill.material.opacity = amount * fillOpacity;
    return amount;
  }
  function dispose() {
    group.removeFromParent();
    geometry.dispose(); edges.geometry.dispose();
    edges.material.dispose(); fill.material.dispose();
  }
  return {group, set, animate, dispose};
}

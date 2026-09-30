import * as THREE from 'three';

// These particles illustrate calculation order, not live model execution.
export function createFlowWire(curve, color = '#7194ba', radius = .075) {
  const group = new THREE.Group();
  const material = new THREE.MeshStandardMaterial({color, roughness: .4});
  const tube = new THREE.Mesh(new THREE.TubeGeometry(curve, 40, radius, 8, false), material);
  group.add(tube);
  const particles = Array.from({length: 4}, () => {
    const particle = new THREE.Mesh(new THREE.SphereGeometry(radius * 1.8, 8, 6), new THREE.MeshBasicMaterial({color: '#e9f8ff'}));
    group.add(particle);
    return particle;
  });
  const arrow = new THREE.Mesh(new THREE.ConeGeometry(radius * 2.4, radius * 5, 10), material);
  group.add(arrow);
  function update(next) {
    curve = next;
    tube.geometry.dispose();
    tube.geometry = new THREE.TubeGeometry(curve, 40, radius, 8, false);
    arrow.position.copy(curve.getPoint(.93));
    arrow.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), curve.getTangent(.93).normalize());
  }
  function animate(now, reduced = false) {
    particles.forEach((particle, index) => {
      particle.position.copy(curve.getPoint((index / particles.length + (reduced ? 0 : now / 3200)) % 1));
    });
  }
  function dispose() {
    tube.geometry.dispose(); arrow.geometry.dispose(); material.dispose();
    particles.forEach(particle => {particle.geometry.dispose(); particle.material.dispose();});
    group.removeFromParent();
  }
  update(curve);
  return {group, particles, update, animate, dispose};
}

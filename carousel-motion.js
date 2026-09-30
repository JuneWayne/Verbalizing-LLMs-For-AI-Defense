// Positions are measured in model slots, so the carousel can cross either end.
export function nearestSlot(index, position, count) {
  return index + Math.round((position - index) / count) * count;
}

// Exact critically damped spring: changing the destination preserves velocity.
export function advanceSpring(position, velocity, target, seconds) {
  const speed = 11;
  const distance = position - target;
  const combined = velocity + speed * distance;
  const decay = Math.exp(-speed * seconds);
  return {position: target + (distance + combined * seconds) * decay, velocity: (velocity - speed * combined * seconds) * decay};
}

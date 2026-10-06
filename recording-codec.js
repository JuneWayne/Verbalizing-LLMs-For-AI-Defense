// Restore the exact recorded doubles. This changes storage, not model scores.
export function unpackRecording(data) {
  if (data.encoding !== 'float64-table-v1') return data;
  const bytes = Uint8Array.from(atob(data.numbers), character => character.charCodeAt(0));
  const numbers = new DataView(bytes.buffer);
  let offset = 0;
  function visit(value) {
    if (value && !Array.isArray(value) && typeof value === 'object' && value.$number === true && Object.keys(value).length === 1) {
      const number = numbers.getFloat64(offset, true);
      offset += 8;
      return number;
    }
    if (Array.isArray(value)) return value.map(visit);
    if (value && typeof value === 'object') {
      return Object.fromEntries(Object.keys(value).sort().map(key => [key, visit(value[key])]));
    }
    return value;
  }
  const result = visit(data.data);
  if (offset !== bytes.length) throw new Error('The recording number count does not match.');
  return result;
}

// Keep long responses complete while holding only a few numerical chunks in memory.
export function createFrameLoader(recording, read, signal) {
  const summaries = recording.events.filter(event => event.type === 'token');
  const frames = summaries.slice();
  const chunks = recording.frame_chunks || [];
  const loaded = new Map();
  const pending = new Map();
  let activeStart = null;
  const chunkAt = index => chunks.find(row => index >= row.start && index < row.start + row.count);

  function focus(index) {
    const chunk = chunkAt(index);
    activeStart = chunk?.start ?? null;
    // Dragging must not leave old downloads competing with the selected step.
    for (const [start, entry] of pending) {
      if (start !== activeStart && start !== chunk?.start + chunk?.count) {
        pending.delete(start);
        entry.controller.abort();
      }
    }
    if (loaded.has(activeStart)) {
      const active = loaded.get(activeStart);
      loaded.delete(activeStart);
      loaded.set(activeStart, active);
    }
  }

  async function ensure(index) {
    if (!frames[index]) throw new Error('This response token is missing.');
    if (frames[index].layer_values) return frames[index];
    const chunk = chunkAt(index);
    if (!chunk) throw new Error('This response chunk is missing.');
    if (!pending.has(chunk.start)) {
      const controller = new AbortController();
      const abort = () => controller.abort();
      if (signal?.aborted) abort();
      signal?.addEventListener('abort', abort, {once: true});
      const entry = {controller};
      entry.promise = read(chunk.file, chunk.sha256, controller.signal).then(saved => {
        if (controller.signal.aborted) throw new DOMException('Outdated response step', 'AbortError');
        if (!Array.isArray(saved.frames) || saved.frames.length !== chunk.count) throw new Error('Incomplete response chunk.');
        saved.frames.forEach((frame, offset) => {
          const summary = summaries[chunk.start + offset];
          if (frame.step !== summary.step || frame.token_id !== summary.token_id || frame.text !== summary.text || !frame.layer_values) throw new Error('Response chunk does not match its recording.');
        });
        saved.frames.forEach((frame, offset) => { frames[chunk.start + offset] = frame; });
        loaded.set(chunk.start, chunk);
        // Never evict the selected chunk when an earlier download finishes late.
        while (loaded.size > 3) {
          const oldest = [...loaded.values()].find(row => row.start !== activeStart);
          loaded.delete(oldest.start);
          for (let offset = 0; offset < oldest.count; offset++) frames[oldest.start + offset] = summaries[oldest.start + offset];
        }
        return saved.frames;
      }).finally(() => {
        signal?.removeEventListener('abort', abort);
        if (pending.get(chunk.start) === entry) pending.delete(chunk.start);
      });
      pending.set(chunk.start, entry);
    }
    const saved = await pending.get(chunk.start).promise;
    return saved[index - chunk.start];
  }

  function prefetch(index) {
    const chunk = chunkAt(index);
    // Start the next chunk early, rather than waiting until playback nearly reaches it.
    if (chunk && chunk.start + chunk.count < frames.length) {
      ensure(chunk.start + chunk.count).catch(() => {});
    }
  }
  return {frames, summaries, ensure, focus, prefetch, get loadedChunks() { return loaded.size; }};
}

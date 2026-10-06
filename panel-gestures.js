// Native touch scrolling, touchpad movement, and mouse dragging share one viewport.
export function connectPanelGestures(viewport, isDetail) {
  let drag = null, suppressClick = false;
  viewport.addEventListener('wheel', event => {
    if (isDetail() || event.ctrlKey || !(Math.abs(event.deltaX) > Math.abs(event.deltaY) || event.shiftKey)) return;
    event.preventDefault(); event.stopPropagation();
    const unit = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? viewport.clientWidth : 1;
    viewport.scrollLeft += (event.deltaX || event.deltaY) * unit;
  }, {capture: true, passive: false});
  viewport.addEventListener('pointerdown', event => {
    suppressClick = false;
    if (isDetail() || event.pointerType === 'touch' || event.button !== 0 || event.target.closest('select,input,textarea,a')) return;
    drag = {id: event.pointerId, x: event.clientX, y: event.clientY, left: viewport.scrollLeft, moved: false};
  }, true);
  viewport.addEventListener('pointermove', event => {
    if (!drag || drag.id !== event.pointerId) return;
    const dx = event.clientX - drag.x, dy = event.clientY - drag.y;
    if (!drag.moved && (Math.abs(dx) < 5 || Math.abs(dx) < Math.abs(dy))) return;
    drag.moved = true; suppressClick = true;
    viewport.setPointerCapture(event.pointerId);
    viewport.classList.add('dragging');
    viewport.scrollLeft = drag.left - dx / (Number(getComputedStyle(document.body).zoom) || 1);
    event.preventDefault(); event.stopPropagation();
  }, true);
  const finish = () => { drag = null; viewport.classList.remove('dragging'); };
  viewport.addEventListener('pointerup', finish, true);
  viewport.addEventListener('pointercancel', finish, true);
  viewport.addEventListener('lostpointercapture', finish);
  viewport.addEventListener('click', event => {
    if (!suppressClick) return;
    suppressClick = false; event.preventDefault(); event.stopImmediatePropagation();
  }, true);
  viewport.addEventListener('keydown', event => {
    if (isDetail() || event.target !== viewport) return;
    const delta = event.key === 'ArrowLeft' ? -100 : event.key === 'ArrowRight' ? 100 : 0;
    if (delta) { event.preventDefault(); viewport.scrollLeft += delta; }
  });
}

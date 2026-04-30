// tooltips.js
// Tooltip display logic; tooltip text lives as data-tooltip attributes in the HTML.
export function initTooltips() {
    const box = document.createElement('div');
    box.className = 'tooltip-box';
    document.body.appendChild(box);

    let hideTimer = null;
    let showTimer = null;

    function show(text, el) {
        clearTimeout(hideTimer);
        clearTimeout(showTimer);
        showTimer = setTimeout(() => {
            box.textContent = text;
            box.style.visibility = 'hidden';
            box.style.display    = 'block';
            position(el);
            box.style.visibility = 'visible';
        }, 1000);
    }

    function position(el) {
        const r  = el.getBoundingClientRect();
        const bw = box.offsetWidth, bh = box.offsetHeight;
        let left = r.left + r.width / 2 - bw / 2 + 10;
        let top  = r.top + window.scrollY - bh - 14;
        left = Math.max(6, Math.min(left, window.innerWidth - bw - 6));
        if (top < window.scrollY + 6) top = r.bottom + window.scrollY + 8;
        box.style.left = left + 'px';
        box.style.top  = top  + 'px';
    }

    function hide() {
        clearTimeout(showTimer);
        hideTimer = setTimeout(() => { box.style.display = 'none'; }, 60);
    }

    function anchor(el) {
        if (el.tagName === 'BUTTON') return el;
        const row = el.closest('.ctrl-row, .slider-row');
        if (row) return row.querySelector('.ctrl-label, .slider-header') ?? el;
        return el.closest('label') ?? el;
    }

    for (const el of document.querySelectorAll('[data-tooltip]')) {
        const text = el.dataset.tooltip;
        const a = anchor(el);
        a.addEventListener('mouseenter', () => show(text, a));
        a.addEventListener('mousemove',  () => position(a));
        a.addEventListener('mouseleave', hide);
    }
}

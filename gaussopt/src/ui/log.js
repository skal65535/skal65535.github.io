// log.js — The page's one log: timestamped lines tagged by source ('app', 'sfm', 'train') and
// level ('info', 'ok', 'warn', 'error'), repeats collapsed into a ×N counter, the last MAX_LINES
// kept. The filter shows one source, or the warnings and errors of all of them.
"use strict";

const MAX_LINES = 500;
const LOG_FILTERS = { all: 'all', app: 'app', 'video → cameras': 'sfm', training: 'train', '⚠ problems': 'warn' };

export class LogPanel {
  // body: the scrolling element of the lines; filter: a <select> (optional)
  constructor(body, filter = null) {
    this.body = body;
    if (filter) {
      for (const [label, value] of Object.entries(LOG_FILTERS)) filter.add(new Option(label, value));
      filter.addEventListener('change', () => { this.body.dataset.filter = filter.value; this._toBottom(); });
    }
  }

  add(msg, level = 'info', source = 'app') {
    const body = this.body, last = body.lastElementChild;
    if (last && last._msg === msg && last._level === level && last._source === source) {
      last.querySelector('.log-count').textContent = `×${++last._count}`;
      return;
    }
    const atBottom = body.scrollHeight - body.scrollTop - body.clientHeight < 4;
    const line = document.createElement('div');
    line.className = `log-line log-${level} log-src-${source}`;
    Object.assign(line, { _msg: msg, _level: level, _source: source, _count: 1 });
    const t = document.createElement('span');
    t.className = 'log-time';
    t.textContent = new Date().toTimeString().slice(0, 8);
    const c = document.createElement('span');
    c.className = 'log-count';
    line.append(t, document.createTextNode(msg), c);
    body.appendChild(line);
    while (body.childElementCount > MAX_LINES) body.firstElementChild.remove();
    if (atBottom) this._toBottom();
  }

  clear() { this.body.textContent = ''; }

  // The messages of one source (all when null), one per line
  text(source = null) {
    return [...this.body.children].filter(l => !source || l._source === source).map(l => l._msg).join('\n');
  }

  _toBottom() { this.body.scrollTop = this.body.scrollHeight; }
}

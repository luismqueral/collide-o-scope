// matrix.js — transposed parameter matrix view.
//
// Rows = parameters (from matrix-schema.js), columns = channels:
//   MASTER (effects)  ·  VHS (ntsc)  ·  one column per layer.
//
// Reuses app.js globals: sendAction, makeValueEditable, openEditor, formatValue.
// All cell lookups go through `cellIndex` (keyed "<colKey>|<paramKey>") so the
// duplicated param keys across columns never collide.

(function () {
  'use strict';

  const groups = window.MATRIX_GROUPS;
  const applies = window.CHANNEL_APPLIES;

  // --- DOM handles ---
  const app = document.querySelector('.app');
  const gridEl = document.getElementById('matrix-grid');
  const scrollEl = document.getElementById('mx-scroll');

  // --- State ---
  let view = 'classic';
  let built = false;
  let colSig = '';
  let lastMsg = null;
  let columns = [];                 // [{ key, kind, label, index? }]
  let rows = [];                    // [{ group, def, cells: [cellInfo|null] }]
  const cellIndex = new Map();      // "<colKey>|<paramKey>" -> cellInfo
  const collapsed = new Set();      // collapsed group names
  let navRows = [];                 // rows visible for keyboard nav
  let focus = { r: 0, c: 0 };       // logical focus into navRows / columns

  // cellInfo = { el, valEl, fxBtn, def, col }

  // =====================================================================
  // Column model
  // =====================================================================
  function buildColumns(msg) {
    const cols = [
      { key: 'master', kind: 'master', label: 'MASTER' },
    ];
    (msg.layers || []).forEach((l, i) => {
      cols.push({ key: 'layer:' + l.id, kind: 'layer', label: 'L' + (i + 1), index: i, id: l.id, filename: l.filename });
    });
    return cols;
  }

  function computeSig(msg) {
    return (msg.layers || []).map((l) => l.id).join(',');
  }

  // =====================================================================
  // Value access
  // =====================================================================
  function snapshotKey(def) {
    return def.key === 'clip' ? 'filename' : def.key;
  }

  function readValue(msg, col, def) {
    const k = snapshotKey(def);
    if (col.kind === 'master') {
      // VHS params live under Master but read from the separate ntsc namespace.
      if (def.channels === 'ntsc') return msg.ntsc ? msg.ntsc[k] : undefined;
      return msg.effects ? msg.effects[k] : undefined;
    }
    const layer = msg.layers && msg.layers[col.index];
    return layer ? layer[k] : undefined;
  }

  function readAutos(msg, col) {
    if (col.kind === 'master') return [msg.automations || {}, msg.automation_errors || {}];
    if (col.kind === 'layer') {
      const layer = msg.layers && msg.layers[col.index];
      return [layer ? (layer.automations || {}) : {}, layer ? (layer.automation_errors || {}) : {}];
    }
    return [{}, {}];
  }

  // =====================================================================
  // Action routing per column kind
  // =====================================================================
  function setValueAction(col, def, value) {
    if (col.kind === 'master') {
      if (def.channels === 'ntsc') return { action: 'set_ntsc_param', param: def.key, value };
      return { action: 'set_param', param: def.key, value };
    }
    return { action: 'set_layer_param', index: col.index, param: def.key, value };
  }

  function editCtx(col, def) {
    // VHS/NTSC params have no automation channel even though they sit in the
    // Master column — typing a formula into one is a no-op.
    const ntsc = def.channels === 'ntsc';
    return {
      setValue: (v) => sendAction(setValueAction(col, def, v)),
      setAutomation: (expr) => {
        if (ntsc) return;
        if (col.kind === 'master') sendAction({ action: 'set_automation', param: def.key, expr });
        else if (col.kind === 'layer') sendAction({ action: 'set_layer_automation', index: col.index, param: def.key, expr });
      },
      clearAutomation: () => {
        if (ntsc) return;
        if (col.kind === 'master') sendAction({ action: 'clear_automation', param: def.key });
        else if (col.kind === 'layer') sendAction({ action: 'clear_layer_automation', index: col.index, param: def.key });
      },
    };
  }

  // =====================================================================
  // Build
  // =====================================================================
  function buildMatrix(msg) {
    columns = buildColumns(msg);
    rows = [];
    cellIndex.clear();
    gridEl.innerHTML = '';
    gridEl.style.setProperty('--mx-cols', columns.length);

    // Header row: corner + column heads
    const corner = document.createElement('div');
    corner.className = 'mx-corner';
    corner.textContent = '';
    gridEl.appendChild(corner);
    columns.forEach((col) => {
      const h = document.createElement('div');
      h.className = 'mx-colhead mx-col-' + col.kind;
      h.textContent = col.label;
      if (col.filename) h.title = col.filename;
      gridEl.appendChild(h);
    });

    // Group + param rows
    groups.forEach((group) => {
      // Group header row (spans all columns, click toggles collapse)
      const gh = document.createElement('div');
      gh.className = 'mx-group';
      gh.dataset.group = group.name;
      gh.innerHTML = '<span class="mx-chevron">\u25BC</span><span class="mx-group-label">' + group.name + '</span>';
      gh.addEventListener('click', () => toggleGroup(group.name));
      gridEl.appendChild(gh);

      group.params.forEach((def) => {
        const rowObj = { group: group.name, def, cells: [] };

        const label = document.createElement('div');
        label.className = 'mx-label';
        label.dataset.group = group.name;
        label.textContent = def.label;
        gridEl.appendChild(label);
        rowObj.labelEl = label;

        columns.forEach((col, c) => {
          if (!applies(def, col.kind)) {
            const na = document.createElement('div');
            na.className = 'mx-cell mx-na';
            na.dataset.group = group.name;
            na.textContent = '\u2014';
            gridEl.appendChild(na);
            rowObj.cells[c] = null;
            return;
          }
          const info = buildCell(def, col, group.name);
          gridEl.appendChild(info.el);
          rowObj.cells[c] = info;
          cellIndex.set(col.key + '|' + def.key, info);
        });

        rows.push(rowObj);
      });
    });

    // Re-apply any collapsed groups to the freshly built cells.
    collapsed.forEach((name) => applyCollapse(name, true));

    built = true;
    rebuildNav();
  }

  function buildCell(def, col, groupName) {
    const el = document.createElement('div');
    el.className = 'mx-cell mx-ptype-' + def.ptype;
    el.dataset.group = groupName;
    el.dataset.col = col.key;
    el.dataset.param = def.key;

    const info = { el, def, col, valEl: null, fxBtn: null, value: undefined, colorInput: null };

    if (def.ptype === 'float' || def.ptype === 'bipolar') {
      const bar = document.createElement('div');
      bar.className = 'mx-bar';
      const fill = document.createElement('div');
      fill.className = 'mx-bar-fill';
      bar.appendChild(fill);
      info.fill = fill;

      const val = document.createElement('span');
      val.className = 'mx-val';
      bar.appendChild(val);
      info.valEl = val;

      el.appendChild(bar);

      const ctx = editCtx(col, def);
      info.ctx = ctx;
      makeValueEditable(val, ctx);
      attachScrub(bar, info);

      if (def.automatable) {
        const fx = document.createElement('button');
        fx.className = 'mx-fx';
        fx.type = 'button';
        fx.textContent = '\u0192';
        fx.title = 'Automate this parameter';
        fx.addEventListener('click', (e) => {
          e.stopPropagation();
          openEditor({
            scope: col.kind === 'master' ? 'master' : 'layer',
            index: col.index,
            param: def.key,
            lo: def.min, hi: def.max, step: def.step,
            label: def.label,
          }, val);
        });
        el.appendChild(fx);
        info.fxBtn = fx;
      }
    } else if (def.ptype === 'bool') {
      const val = document.createElement('span');
      val.className = 'mx-val mx-bool';
      val.textContent = '\u25CB';
      el.appendChild(val);
      info.valEl = val;
      el.addEventListener('click', () => toggleBool(info));
    } else if (def.ptype === 'enum') {
      const val = document.createElement('span');
      val.className = 'mx-val mx-enum';
      el.appendChild(val);
      info.valEl = val;
      el.addEventListener('click', () => cycleEnum(info, 1));
    } else if (def.ptype === 'color') {
      const input = document.createElement('input');
      input.type = 'color';
      input.className = 'mx-color';
      input.addEventListener('input', () => {
        sendAction(setValueAction(col, def, input.value));
      });
      el.appendChild(input);
      info.colorInput = input;
      info.valEl = input;
    } else if (def.ptype === 'clip') {
      const val = document.createElement('span');
      val.className = 'mx-val mx-clip';
      el.appendChild(val);
      info.valEl = val;
    }

    el.addEventListener('mousedown', () => focusCell(info));
    return info;
  }

  // Drag horizontally across a float/bipolar bar to scrub its value. The bar is
  // the interaction surface (the value text has pointer-events:none), so a clean
  // click — negligible movement — falls through to the text editor instead.
  function attachScrub(bar, info) {
    bar.addEventListener('mousedown', (e) => {
      if (e.button !== 0) return;
      // If a live text input is already open in this cell, leave it alone.
      if (info.valEl && info.valEl.querySelector && info.valEl.querySelector('input')) return;
      const { def } = info;
      const span = (def.max - def.min) || 1;
      const rect = bar.getBoundingClientRect();
      const startX = e.clientX;
      const startVal = typeof info.value === 'number' ? info.value : def.def;
      let moved = false;
      info.scrubbing = true;
      focusCell(info);

      // Scrubbing is a manual edit; drop any running automation first.
      if (info.valEl && info.valEl.dataset.expr && info.ctx) info.ctx.clearAutomation();

      const onMove = (ev) => {
        const dx = ev.clientX - startX;
        if (Math.abs(dx) > 2) moved = true;
        let v = startVal + (dx / rect.width) * span;
        if (def.step) v = Math.round(v / def.step) * def.step;
        v = Math.max(def.min, Math.min(def.max, v));
        info.value = v;
        if (info.valEl) info.valEl.textContent = formatValue(v, def.min, def.max, def.step);
        paintBar(info, v);
        markChanged(info, !nearlyEqual(v, def.def));
        sendAction(setValueAction(info.col, def, v));
      };

      const onUp = () => {
        document.removeEventListener('mousemove', onMove);
        document.removeEventListener('mouseup', onUp);
        info.scrubbing = false;
        // A clean click (no real drag) opens the text editor.
        if (!moved && info.valEl) info.valEl.click();
      };

      document.addEventListener('mousemove', onMove);
      document.addEventListener('mouseup', onUp);
      e.preventDefault();
    });
  }

  // =====================================================================
  // Sync
  // =====================================================================
  function syncMatrix(msg) {
    lastMsg = msg;
    if (view !== 'matrix') return;
    ensureBuilt(msg);
    updateCells(msg);
    if (drawerOpen) renderDrawers(msg);
  }

  function ensureBuilt(msg) {
    const sig = computeSig(msg);
    if (!built || sig !== colSig) {
      const keepFocus = { r: focus.r, c: focus.c };
      buildMatrix(msg);
      colSig = sig;
      focus = { r: Math.min(keepFocus.r, navRows.length - 1), c: keepFocus.c };
      if (focus.r < 0) focus = { r: 0, c: 0 };
      ensureFocusable();
      paintFocus(false);
    }
  }

  // Move logical focus onto the first applicable cell if it currently points at
  // a non-applicable (null) slot.
  function ensureFocusable() {
    if (navInfo(focus.r, focus.c)) return;
    for (let r = 0; r < navRows.length; r++) {
      const c = navRows[r].cells.findIndex(Boolean);
      if (c >= 0) { focus = { r, c }; return; }
    }
  }

  function updateCells(msg) {
    for (let ci = 0; ci < columns.length; ci++) {
      const col = columns[ci];
      const [autos, errors] = readAutos(msg, col);
      for (let ri = 0; ri < rows.length; ri++) {
        const info = rows[ri].cells[ci];
        if (!info) continue;
        updateCell(info, msg, autos, errors);
      }
    }
  }

  function updateCell(info, msg, autos, errors) {
    const { def, col, valEl } = info;
    const v = readValue(msg, col, def);
    info.value = v;

    // Frozen-while-editing guard: don't clobber a live text input or an
    // active color picker.
    const editing = valEl && valEl.querySelector && valEl.querySelector('input');
    const colorActive = info.colorInput && document.activeElement === info.colorInput;
    if (editing || colorActive || info.scrubbing) {
      applyMatrixAutoState(info, autos, errors);
      return;
    }

    if (def.ptype === 'float' || def.ptype === 'bipolar') {
      const num = typeof v === 'number' ? v : 0;
      valEl.textContent = formatValue(num, def.min, def.max, def.step);
      paintBar(info, num);
      markChanged(info, !nearlyEqual(num, def.def));
    } else if (def.ptype === 'bool') {
      const on = !!v;
      valEl.textContent = on ? '\u25CF' : '\u25CB';
      info.el.classList.toggle('mx-on', on);
      markChanged(info, on !== def.def);
    } else if (def.ptype === 'enum') {
      const opt = (def.options || []).find((o) => o.value === v);
      valEl.textContent = opt ? opt.label : String(v);
      markChanged(info, v !== def.def);
    } else if (def.ptype === 'color') {
      if (typeof v === 'string' && v) info.colorInput.value = v;
      markChanged(info, (v || '').toLowerCase() !== (def.def || '').toLowerCase());
    } else if (def.ptype === 'clip') {
      const name = typeof v === 'string' ? v.replace(/\.[^.]+$/, '') : '';
      valEl.textContent = name;
      valEl.title = v || '';
    }

    applyMatrixAutoState(info, autos, errors);
  }

  // Matrix's own automation-state painter (app.js's applyAutomationState is
  // coupled to .param-row, so we paint here directly on the cell).
  function applyMatrixAutoState(info, autos, errors) {
    if (!info.valEl) return;
    const key = info.def.key;
    const expr = autos && autos[key];
    const err = errors && errors[key];
    if (expr) {
      info.el.classList.add('mx-automated');
      info.valEl.dataset.expr = expr;
    } else {
      info.el.classList.remove('mx-automated');
      delete info.valEl.dataset.expr;
      delete info.valEl.dataset.aectl;
    }
    if (info.fxBtn) info.fxBtn.classList.toggle('active', !!expr);
    if (err) {
      info.el.classList.add('mx-error');
      info.el.title = err;
    } else {
      info.el.classList.remove('mx-error');
      if (info.def.ptype !== 'clip') info.el.removeAttribute('title');
    }
  }

  function paintBar(info, v) {
    if (!info.fill) return;
    const { min, max, ptype } = info.def;
    const span = max - min || 1;
    if (ptype === 'bipolar') {
      const origin = ((0 - min) / span) * 100;
      const pos = ((v - min) / span) * 100;
      const lo = Math.min(origin, pos);
      const hi = Math.max(origin, pos);
      info.fill.style.left = lo + '%';
      info.fill.style.width = (hi - lo) + '%';
    } else {
      const pct = ((v - min) / span) * 100;
      info.fill.style.left = '0%';
      info.fill.style.width = Math.max(0, Math.min(100, pct)) + '%';
    }
  }

  function markChanged(info, changed) {
    info.el.classList.toggle('mx-changed', !!changed);
  }

  function nearlyEqual(a, b) {
    if (typeof b !== 'number') return false;
    return Math.abs(a - b) < 1e-4;
  }

  // =====================================================================
  // Interactions (bool / enum)
  // =====================================================================
  function toggleBool(info) {
    focusCell(info);
    const { def, col } = info;
    const cur = !!info.value;
    if (col.kind === 'layer' && def.key === 'visible') {
      sendAction({ action: 'toggle_visibility', index: col.index });
    } else if (col.kind === 'layer' && def.key === 'paused') {
      sendAction({ action: 'toggle_layer_pause', index: col.index });
    } else {
      sendAction(setValueAction(col, def, !cur));
    }
  }

  function cycleEnum(info, dir) {
    focusCell(info);
    const { def, col } = info;
    const opts = def.options || [];
    if (!opts.length) return;
    let i = opts.findIndex((o) => o.value === info.value);
    if (i < 0) i = 0;
    const next = opts[(i + dir + opts.length) % opts.length];
    sendAction(setValueAction(col, def, next.value));
  }

  // =====================================================================
  // Keyboard navigation
  // =====================================================================
  function rebuildNav() {
    navRows = rows.filter((r) => !collapsed.has(r.group));
    if (focus.r >= navRows.length) focus.r = Math.max(0, navRows.length - 1);
  }

  function navInfo(r, c) {
    const row = navRows[r];
    if (!row) return null;
    return row.cells[c] || null;
  }

  function focusCell(info) {
    // Map an info back to (r,c) in navRows
    for (let r = 0; r < navRows.length; r++) {
      const c = navRows[r].cells.indexOf(info);
      if (c >= 0) { focus = { r, c }; paintFocus(true); return; }
    }
  }

  function paintFocus(scroll) {
    gridEl.querySelectorAll('.mx-focus').forEach((e) => e.classList.remove('mx-focus'));
    const info = navInfo(focus.r, focus.c);
    if (info) {
      info.el.classList.add('mx-focus');
      if (scroll) info.el.scrollIntoView({ block: 'nearest', inline: 'nearest' });
    }
  }

  function moveFocus(dr, dc) {
    if (!navRows.length) return;
    let { r, c } = focus;
    if (dc !== 0) {
      let nc = c;
      for (let i = 0; i < columns.length; i++) {
        nc = (nc + dc + columns.length) % columns.length;
        if (navInfo(r, nc)) { c = nc; break; }
      }
    }
    if (dr !== 0) {
      let nr = r;
      for (let i = 0; i < navRows.length; i++) {
        nr = (nr + dr + navRows.length) % navRows.length;
        if (navInfo(nr, c)) { r = nr; break; }
      }
    }
    focus = { r, c };
    paintFocus(true);
  }

  function activateFocus() {
    const info = navInfo(focus.r, focus.c);
    if (!info) return;
    const { def } = info;
    if (def.ptype === 'float' || def.ptype === 'bipolar') {
      info.valEl.click(); // makeValueEditable -> text input
    } else if (def.ptype === 'bool') {
      toggleBool(info);
    } else if (def.ptype === 'enum') {
      cycleEnum(info, 1);
    } else if (def.ptype === 'color') {
      info.colorInput.click();
    }
  }

  // Tab: apply the focused param's current value to every layer column it
  // applies to.
  function applyAcrossLayers() {
    const info = navInfo(focus.r, focus.c);
    if (!info) return;
    const def = info.def;
    if (!applies(def, 'layer')) return;
    const v = info.value;
    if (v === undefined) return;
    columns.forEach((col) => {
      if (col.kind !== 'layer') return;
      if (col === info.col) return;
      sendAction(setValueAction(col, def, v));
    });
    flashRow(info);
  }

  function flashRow(info) {
    const row = navRows[focus.r];
    if (!row) return;
    row.cells.forEach((c) => { if (c) { c.el.classList.add('mx-flash'); } });
    setTimeout(() => row.cells.forEach((c) => { if (c) c.el.classList.remove('mx-flash'); }), 220);
  }

  function onKey(e) {
    if (view !== 'matrix') return;
    // Let live inputs/selects handle their own keys.
    const ae = document.activeElement;
    if (ae && (ae.tagName === 'INPUT' || ae.tagName === 'SELECT' || ae.tagName === 'TEXTAREA')) return;
    // Don't fight the automation modal.
    const modal = document.getElementById('automation');
    if (modal && !modal.hasAttribute('hidden')) return;

    switch (e.key) {
      case 'ArrowLeft': case 'h': moveFocus(0, -1); e.preventDefault(); break;
      case 'ArrowRight': case 'l': moveFocus(0, 1); e.preventDefault(); break;
      case 'ArrowUp': case 'k': moveFocus(-1, 0); e.preventDefault(); break;
      case 'ArrowDown': case 'j': moveFocus(1, 0); e.preventDefault(); break;
      case 'Enter': activateFocus(); e.preventDefault(); break;
      case ' ': {
        const info = navInfo(focus.r, focus.c);
        if (info && info.def.ptype === 'bool') { toggleBool(info); e.preventDefault(); }
        break;
      }
      case 'Tab': applyAcrossLayers(); e.preventDefault(); break;
    }
  }

  // =====================================================================
  // Collapsible groups
  // =====================================================================
  function toggleGroup(name) {
    if (collapsed.has(name)) collapsed.delete(name);
    else collapsed.add(name);
    applyCollapse(name, collapsed.has(name));
    rebuildNav();
    ensureFocusable();
    paintFocus(false);
  }

  function applyCollapse(name, isCollapsed) {
    gridEl.querySelectorAll('[data-group="' + cssEscape(name) + '"]').forEach((el) => {
      if (el.classList.contains('mx-group')) {
        el.classList.toggle('mx-collapsed', isCollapsed);
      } else {
        el.style.display = isCollapsed ? 'none' : '';
      }
    });
  }

  function cssEscape(s) {
    return (window.CSS && CSS.escape) ? CSS.escape(s) : s.replace(/["\\]/g, '\\$&');
  }

  // =====================================================================
  // View toggle + drawers
  // =====================================================================
  function setView(next) {
    view = next;
    app.classList.toggle('view-matrix', next === 'matrix');
    app.classList.toggle('view-classic', next === 'classic');
    document.getElementById('view-classic-btn').classList.toggle('is-active', next === 'classic');
    document.getElementById('view-matrix-btn').classList.toggle('is-active', next === 'matrix');
    if (next === 'matrix' && lastMsg) {
      ensureBuilt(lastMsg);
      updateCells(lastMsg);
      if (drawerOpen) renderDrawers(lastMsg);
    }
  }

  let drawerOpen = null; // 'patches' | 'library' | null
  function toggleDrawer(which) {
    drawerOpen = drawerOpen === which ? null : which;
    document.getElementById('mx-drawer-patches').classList.toggle('open', drawerOpen === 'patches');
    document.getElementById('mx-drawer-library').classList.toggle('open', drawerOpen === 'library');
    document.getElementById('drawer-patches-btn').classList.toggle('is-active', drawerOpen === 'patches');
    document.getElementById('drawer-library-btn').classList.toggle('is-active', drawerOpen === 'library');
    if (drawerOpen && lastMsg) renderDrawers(lastMsg);
  }

  function renderDrawers(msg) {
    if (drawerOpen === 'patches') renderPatches(msg.patches || []);
    if (drawerOpen === 'library') renderLibrary(msg.library || []);
  }

  function renderPatches(patches) {
    const list = document.getElementById('mx-patches-list');
    const sig = patches.join('\u0001');
    if (list.dataset.sig === sig) return;
    list.dataset.sig = sig;
    list.innerHTML = '';
    patches.forEach((name) => {
      const row = document.createElement('div');
      row.className = 'mx-patch-row';
      const load = document.createElement('button');
      load.className = 'mx-patch-load';
      load.textContent = name;
      load.title = 'Load patch';
      load.addEventListener('click', () => sendAction({ action: 'load_patch', name }));
      const del = document.createElement('button');
      del.className = 'mx-patch-del';
      del.textContent = '\u00D7';
      del.title = 'Delete patch';
      del.addEventListener('click', () => sendAction({ action: 'delete_patch', name }));
      row.appendChild(load);
      row.appendChild(del);
      list.appendChild(row);
    });
  }

  function renderLibrary(library) {
    const grid = document.getElementById('mx-library-grid');
    const sig = library.join('\u0001');
    if (grid.dataset.sig === sig) return;
    grid.dataset.sig = sig;
    grid.innerHTML = '';
    library.forEach((filename) => {
      const item = document.createElement('div');
      item.className = 'mx-lib-item';
      item.textContent = filename.replace(/\.[^.]+$/, '');
      item.title = filename + ' (double-click to add)';
      item.addEventListener('dblclick', () => sendAction({ action: 'add_layer', filename }));
      grid.appendChild(item);
    });
  }

  // =====================================================================
  // Wiring
  // =====================================================================
  function init() {
    document.getElementById('view-classic-btn').addEventListener('click', () => setView('classic'));
    document.getElementById('view-matrix-btn').addEventListener('click', () => setView('matrix'));
    document.getElementById('drawer-patches-btn').addEventListener('click', () => toggleDrawer('patches'));
    document.getElementById('drawer-library-btn').addEventListener('click', () => toggleDrawer('library'));

    const saveBtn = document.getElementById('mx-patch-save');
    if (saveBtn) saveBtn.addEventListener('click', () => {
      const input = document.getElementById('mx-patch-name');
      const name = (input.value || '').trim();
      if (name) { sendAction({ action: 'save_patch', name }); input.value = ''; }
    });

    document.addEventListener('keydown', onKey);

    window.onMatrixState = syncMatrix;
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();

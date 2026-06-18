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
  const masterGridEl = document.getElementById('mx-master-grid');

  // MASTER is its own single-column panel (a right-side slideout), not an inline
  // grid column. The main grid holds only layer columns.
  const MASTER_COL = { key: 'master', kind: 'master', label: 'MASTER' };

  // --- State ---
  let view = 'matrix'; // default panel; `?view=classic` boots the legacy view
  let built = false;
  let masterBuilt = false;
  let colSig = '';
  let lastMsg = null;
  let columns = [];                 // layer columns only: [{ key, kind, label, index }]
  let rows = [];                    // [{ group, def, cells: [cellInfo|null] }]
  let masterRows = [];              // [{ group, def, cell }] for the master panel
  let outputCells = [];             // [{ refresh }] OUTPUT-section cycle cells
  let vhsPresetName = null;         // currently applied VHS preset name (or null)
  let vhsPresetEls = null;          // { name, mod } spans on the VHS header
  let masterMeterFill = null;       // master output level meter fill (live peak)
  let addColCells = [];             // every cell in the trailing add-layer column
  const cellIndex = new Map();      // "<colKey>|<paramKey>" -> cellInfo
  const collapsed = new Set();      // collapsed group names
  let navRows = [];                 // rows visible for keyboard nav
  let focus = { r: 0, c: 0 };       // logical focus into navRows / columns

  // cellInfo = { el, valEl, fxBtn, def, col }

  // =====================================================================
  // Column model
  // =====================================================================
  function buildColumns(msg) {
    const layers = msg.layers || [];
    // Null state: no layer columns at all. The trailing "+" add column is the
    // entry point for creating the first layer (opens the library in add mode).
    return layers.map((l, i) => ({
      key: 'layer:' + l.id, kind: 'layer', label: 'L' + (i + 1),
      index: i, id: l.id, filename: l.filename,
    }));
  }

  // Does any param in this group apply to the given column kind? Used to drop
  // empty groups (e.g. ANALOG/MOTION have no layer params, so they never appear
  // in the layer grid; LAYER/WARP/etc. have no master params).
  function groupApplies(group, kind) {
    return group.params.some((def) => applies(def, kind));
  }

  // Layer-grid grouping is decoupled from the schema: window.LAYER_GROUPS lists
  // the layer view's own groups/order by param key (SOURCE/BLEND/.../GLITCH),
  // while the master panel keeps iterating `groups` (MATRIX_GROUPS) directly.
  // We resolve each key to its shared ParamDef via defByKey so min/max/step/etc.
  // stay single-sourced in the schema.
  const layerLayout = window.LAYER_GROUPS;
  const defByKey = new Map();
  groups.forEach((g) => g.params.forEach((d) => defByKey.set(d.key, d)));

  // layerGroupOrder() returns schema-shaped groups ({ name, params }) built from
  // the LAYER_GROUPS layout, so buildMatrix's loop works unchanged.
  function layerGroupOrder() {
    return layerLayout.map((g) => ({
      name: g.name,
      params: g.keys.map((k) => defByKey.get(k)).filter(Boolean),
    }));
  }

  function computeSig(msg) {
    return (msg.layers || []).map((l) => l.id).join(',');
  }

  // =====================================================================
  // Value access
  // =====================================================================
  function snapshotKey(def) {
    if (def.snap) return def.snap; // explicit override (e.g. limiter → master_limiter)
    return def.key === 'clip' ? 'filename' : def.key;
  }

  function readValue(msg, col, def) {
    const k = snapshotKey(def);
    if (col.kind === 'master') {
      // VHS params live under Master but read from the separate ntsc namespace.
      if (def.channels === 'ntsc') return msg.ntsc ? msg.ntsc[k] : undefined;
      // Master audio bus reads top-level snapshot fields (master_volume /
      // master_limiter), not the per-frame effects uniform block.
      if (def.channels === 'master_audio') return msg[k];
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
      // Master volume/limiter have a dedicated action (App-level state, not the
      // effects uniform), so they must NOT go through generic set_param.
      if (def.channels === 'master_audio') return { action: 'set_master_audio_param', param: def.key, value };
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
    addColCells = [];
    cellIndex.clear();
    gridEl.innerHTML = '';
    gridEl.style.setProperty('--mx-cols', columns.length);
    // Null state (no layer columns): --mx-cols becomes 0, which would make the
    // grid template's repeat() `repeat(0, …)` (invalid CSS, voids the template).
    // The .mx-empty class swaps in an explicit 2-track template so the trailing
    // "+" add column renders right after the labels — the populated layout minus
    // the layer columns. See #matrix-grid.mx-empty in style.css.
    gridEl.classList.toggle('mx-empty', columns.length === 0);

    // Header row: corner + column heads + a trailing skinny "add layer" column.
    const corner = document.createElement('div');
    corner.className = 'mx-corner';
    gridEl.appendChild(corner);
    columns.forEach((col) => {
      const h = document.createElement('div');
      h.className = 'mx-colhead mx-col-' + col.kind;
      h.textContent = col.label;
      if (col.filename) h.title = col.filename;
      col.headEl = h;
      // Layer columns are draggable to reorder; MASTER stays pinned first.
      // They also carry a clip-timing progress bar beneath the title and a
      // remove (×) button to drop the layer.
      if (col.kind === 'layer') {
        attachColumnDrag(h, col);

        const rm = document.createElement('button');
        rm.className = 'mx-col-remove';
        rm.type = 'button';
        rm.title = 'Remove ' + col.label;
        rm.innerHTML = '<i data-lucide="x"></i>';
        // Swallow the pointerdown so it never starts a column drag.
        rm.addEventListener('pointerdown', (e) => e.stopPropagation());
        rm.addEventListener('click', (e) => {
          e.stopPropagation();
          sendAction({ action: 'remove_layer', index: col.index });
        });
        h.appendChild(rm);

        const prog = document.createElement('div');
        prog.className = 'mx-colhead-prog';
        const fill = document.createElement('div');
        fill.className = 'mx-colhead-prog-fill';
        prog.appendChild(fill);
        h.appendChild(prog);
        col.progFill = fill;
      }
      gridEl.appendChild(h);
    });
    // Trailing add-layer column header. The whole add column is one big click
    // target (delegated in init); this top cell carries the visible "+".
    const addHead = document.createElement('div');
    addHead.className = 'mx-addcol mx-addcol-head';
    addHead.textContent = '+';
    addHead.title = 'Add a layer';
    addColCells.push(addHead);
    gridEl.appendChild(addHead);

    // Group + param rows (layer params only — master params live in the panel).
    // Uses layerGroupOrder() so AUDIO/AUDIO FX sit right under LAYER.
    layerGroupOrder().forEach((group) => {
      if (!groupApplies(group, 'layer')) return;

      // Group header band. The label cell (chevron + name) toggles collapse;
      // each layer column gets its own control cell with per-layer reset +
      // randomize. These control cells stay visible when the group collapses
      // (see applyCollapse) so the band always exposes the buttons.
      const gh = document.createElement('div');
      gh.className = 'mx-group mx-group-head';
      gh.dataset.group = group.name;
      gh.innerHTML = '<span class="mx-chevron">\u25BC</span><span class="mx-group-label">' + group.name + '</span>';
      gh.addEventListener('click', () => toggleGroup(group.name, 'layer'));
      gridEl.appendChild(gh);

      columns.forEach((col) => {
        const ctl = document.createElement('div');
        ctl.className = 'mx-group-ctl';
        ctl.dataset.group = group.name;
        if (col.kind === 'layer') {
          const rnd = document.createElement('button');
          rnd.className = 'mx-grp-btn';
          rnd.type = 'button';
          rnd.title = 'Randomize ' + group.name + ' on ' + col.label;
          rnd.innerHTML = '<i data-lucide="dices"></i>';
          rnd.addEventListener('click', (e) => {
            e.stopPropagation();
            randomizeLayerGroup(group.name, col.index);
          });
          ctl.appendChild(rnd);

          const rst = document.createElement('button');
          rst.className = 'mx-grp-btn';
          rst.type = 'button';
          rst.title = 'Reset ' + group.name + ' on ' + col.label;
          rst.innerHTML = '<i data-lucide="rotate-ccw"></i>';
          rst.addEventListener('click', (e) => {
            e.stopPropagation();
            resetLayerGroup(group.name, col.index);
          });
          ctl.appendChild(rst);
        }
        gridEl.appendChild(ctl);
      });
      // Trailing filler so the header band fills the add column too.
      const ghTail = document.createElement('div');
      ghTail.className = 'mx-group-ctl mx-addcol';
      ghTail.dataset.group = group.name;
      addColCells.push(ghTail);
      gridEl.appendChild(ghTail);

      group.params.forEach((def) => {
        if (!applies(def, 'layer')) return;
        const rowObj = { group: group.name, def, cells: [] };
        const rowIndex = rows.length;
        // The clip row is taller so its thumbnail is large and legible.
        const tall = def.ptype === 'clip';

        const label = document.createElement('div');
        label.className = 'mx-label' + (tall ? ' mx-row-tall' : '');
        label.dataset.group = group.name;
        label.dataset.mxrow = String(rowIndex);
        label.textContent = def.label;
        gridEl.appendChild(label);
        rowObj.labelEl = label;

        columns.forEach((col, c) => {
          if (!applies(def, col.kind)) {
            const na = document.createElement('div');
            na.className = 'mx-cell mx-na' + (tall ? ' mx-row-tall' : '');
            na.dataset.group = group.name;
            na.dataset.mxrow = String(rowIndex);
            na.textContent = '\u2014';
            gridEl.appendChild(na);
            rowObj.cells[c] = null;
            return;
          }
          const info = buildCell(def, col, group.name);
          info.el.dataset.mxrow = String(rowIndex);
          if (tall) info.el.classList.add('mx-row-tall');
          gridEl.appendChild(info.el);
          rowObj.cells[c] = info;
          cellIndex.set(col.key + '|' + def.key, info);
        });
        // Trailing filler for the add column (kept aligned with the row).
        const pad = document.createElement('div');
        pad.className = 'mx-addcol' + (tall ? ' mx-row-tall' : '');
        pad.dataset.group = group.name;
        pad.dataset.mxrow = String(rowIndex);
        addColCells.push(pad);
        gridEl.appendChild(pad);

        rows.push(rowObj);
      });
    });

    // Re-apply any collapsed layer groups to the freshly built cells.
    collapsed.forEach((key) => {
      if (key.startsWith('layer:')) applyCollapse(key.slice(6), 'layer', true);
    });

    // Render the Lucide glyphs in the freshly built group-control buttons.
    if (window.lucide) window.lucide.createIcons();

    built = true;
    rebuildNav();
  }

  // Layer group name -> reset_layer_group key: lowercase, spaces removed
  // (SOURCE→source, COLOR KEY→colorkey, AUDIO FX→audiofx, GLITCH→glitch, ...).
  // Backend arms in main.rs must match these keys.
  function resetGroupKey(name) {
    return name.toLowerCase().replace(/\s+/g, '');
  }

  function resetLayerGroup(groupName, index) {
    sendAction({ action: 'reset_layer_group', index, group: resetGroupKey(groupName) });
  }

  // Client-side randomize (mirrors classic): only numeric (float/bipolar) layer
  // params get a fresh in-range value; toggles/enums/colors/clips are skipped.
  function randomizeLayerGroup(groupName, index) {
    const layout = layerLayout.find((g) => g.name === groupName);
    if (!layout) return;
    layout.keys.forEach((k) => {
      const def = defByKey.get(k);
      if (!def) return;
      if (!applies(def, 'layer')) return;
      if (def.noRandom) return;
      if (def.ptype !== 'float' && def.ptype !== 'bipolar') return;
      const v = randInRange(def.min, def.max, def.step || 0.01);
      sendAction({ action: 'set_layer_param', index, param: def.key, value: v });
    });
  }

  // Swap a layer's clip to a random library entry (in place, keeps FX).
  function randomClip(index) {
    const lib = (lastMsg && lastMsg.library) || [];
    if (!lib.length) return;
    const filename = lib[Math.floor(Math.random() * lib.length)];
    sendAction({ action: 'set_layer_clip', index, filename });
  }

  // Master FX panel — a single-column matrix (label + MASTER) in the right
  // slideout. Independent of layers, so it's built once. Shows only the groups
  // that have master/global params (COLOR, DIGITAL, ANALOG, MOTION, VHS/NTSC).
  function buildMasterPanel() {
    masterRows = [];
    outputCells = [];
    vhsPresetEls = null;
    masterMeterFill = null;
    masterGridEl.innerHTML = '';
    masterGridEl.style.setProperty('--mx-cols', 1);

    // Single header spanning both columns (label + value), styled like the
    // layer column heads (gray), not the accent color.
    const h = document.createElement('div');
    h.className = 'mx-master-head';
    h.textContent = 'MAIN';
    masterGridEl.appendChild(h);

    // OUTPUT section — global render settings (dimensions + framerate). These
    // aren't per-param effects, so they sit above the effect groups as their own
    // band of click-to-cycle cells.
    buildOutputSection();

    // Section-header hand-off (matches the main layer grid): every group header
    // pins at the same top (just below the MASTER head), so as you scroll the
    // next section scrolls up and covers the previous one — one header fixed at
    // a time. The shared top + z-index come from the base .mx-group rule.
    groups.forEach((group) => {
      if (!groupApplies(group, 'master')) return;

      const gh = document.createElement('div');
      gh.className = 'mx-group';
      gh.dataset.group = group.name;
      gh.innerHTML = '<span class="mx-chevron">\u25BC</span><span class="mx-group-label">' + group.name + '</span>';
      gh.addEventListener('click', () => toggleGroup(group.name, 'master'));

      // Right-aligned controls on the header band: an optional VHS preset cycle,
      // then per-group randomize + reset (mirrors the layer columns' buttons).
      const ctl = document.createElement('span');
      ctl.className = 'mx-grp-btns';
      if (group.name === 'VHS/NTSC') ctl.appendChild(buildVhsPresetControl());

      const rnd = document.createElement('button');
      rnd.className = 'mx-grp-btn';
      rnd.type = 'button';
      rnd.title = 'Randomize ' + group.name;
      rnd.innerHTML = '<i data-lucide="dices"></i>';
      rnd.addEventListener('click', (e) => { e.stopPropagation(); randomizeMasterGroup(group.name); });
      ctl.appendChild(rnd);

      const rst = document.createElement('button');
      rst.className = 'mx-grp-btn';
      rst.type = 'button';
      rst.title = 'Reset ' + group.name;
      rst.innerHTML = '<i data-lucide="rotate-ccw"></i>';
      rst.addEventListener('click', (e) => { e.stopPropagation(); resetMasterGroup(group.name); });
      ctl.appendChild(rst);

      gh.appendChild(ctl);
      masterGridEl.appendChild(gh);

      group.params.forEach((def) => {
        if (!applies(def, 'master')) return;

        const mIdx = masterRows.length;

        const label = document.createElement('div');
        label.className = 'mx-label';
        label.dataset.group = group.name;
        label.dataset.mxrow = String(mIdx);
        label.textContent = def.label;
        masterGridEl.appendChild(label);

        const info = buildCell(def, MASTER_COL, group.name);
        info.el.dataset.mxrow = String(mIdx);
        masterGridEl.appendChild(info.el);
        cellIndex.set(MASTER_COL.key + '|' + def.key, info);
        masterRows.push({ group: group.name, def, cell: info });
      });

      // Live output level meter pinned under the AUDIO group (read-only peak,
      // fed each frame from snapshot.meter — not an editable param row).
      if (group.name === 'AUDIO') buildMasterMeterRow();
    });

    masterBuilt = true;
    if (window.lucide) window.lucide.createIcons();
    collapsed.forEach((key) => {
      if (key.startsWith('master:')) applyCollapse(key.slice(7), 'master', true);
    });
  }

  // ---- OUTPUT section (ratio / quality / framerate) -------------------------
  const OUTPUT_RATIOS = ['16:9', '4:3', '1:1', '9:16', '21:9'];
  const OUTPUT_QUALITIES = [720, 1080, 1440];
  const MASTER_FPS = [30, 15, 10, 7.5, 6];

  function buildOutputSection() {
    const head = document.createElement('div');
    head.className = 'mx-group mx-group-static';
    head.innerHTML = '<span class="mx-group-label">OUTPUT</span>';
    masterGridEl.appendChild(head);

    addOutputRow('ratio',
      () => (lastMsg && lastMsg.output_ratio) || '16:9',
      (dir) => {
        const cur = (lastMsg && lastMsg.output_ratio) || '16:9';
        const i = Math.max(0, OUTPUT_RATIOS.indexOf(cur));
        const ratio = OUTPUT_RATIOS[(i + dir + OUTPUT_RATIOS.length) % OUTPUT_RATIOS.length];
        const quality = (lastMsg && lastMsg.output_quality) || 1080;
        sendAction({ action: 'set_output_size', ratio, quality });
      });

    addOutputRow('quality',
      () => String((lastMsg && lastMsg.output_quality) || 1080) + 'p',
      (dir) => {
        const cur = (lastMsg && lastMsg.output_quality) || 1080;
        const i = Math.max(0, OUTPUT_QUALITIES.indexOf(cur));
        const quality = OUTPUT_QUALITIES[(i + dir + OUTPUT_QUALITIES.length) % OUTPUT_QUALITIES.length];
        const ratio = (lastMsg && lastMsg.output_ratio) || '16:9';
        sendAction({ action: 'set_output_size', ratio, quality });
      });

    addOutputRow('fps',
      () => String((lastMsg && lastMsg.framerate) || 30),
      (dir) => {
        const cur = (lastMsg && lastMsg.framerate) || 30;
        const i = Math.max(0, MASTER_FPS.indexOf(cur));
        const value = MASTER_FPS[(i + dir + MASTER_FPS.length) % MASTER_FPS.length];
        sendAction({ action: 'set_master_framerate', value });
      });
  }

  // A label + click-to-cycle value cell, styled like an enum cell. getText()
  // returns the current display string; onCycle(dir) advances + sends the action.
  function addOutputRow(labelText, getText, onCycle) {
    const label = document.createElement('div');
    label.className = 'mx-label';
    label.textContent = labelText;
    masterGridEl.appendChild(label);

    const el = document.createElement('div');
    el.className = 'mx-cell mx-ptype-enum';
    const val = document.createElement('span');
    val.className = 'mx-val mx-enum';
    val.textContent = getText();
    el.appendChild(val);
    el.addEventListener('click', () => onCycle(1));
    masterGridEl.appendChild(el);

    outputCells.push({ refresh: () => { val.textContent = getText(); } });
  }

  // Read-only master output meter: a label + a thin horizontal bar whose fill
  // tracks the live peak level (0..1) broadcast in snapshot.meter.
  function buildMasterMeterRow() {
    const label = document.createElement('div');
    label.className = 'mx-label';
    label.textContent = 'meter';
    masterGridEl.appendChild(label);

    const cell = document.createElement('div');
    cell.className = 'mx-cell mx-meter-cell';
    const meter = document.createElement('div');
    meter.className = 'audio-meter';
    const fill = document.createElement('div');
    fill.className = 'audio-meter-fill';
    meter.appendChild(fill);
    cell.appendChild(meter);
    masterGridEl.appendChild(cell);
    masterMeterFill = fill;
  }

  // ---- Master group reset / randomize ---------------------------------------
  // Client-side, mirroring the layer band: reset writes each applicable master
  // param back to its schema default; randomize scatters numeric params in-range.
  // Enums/bools/colors are left untouched by randomize (matches classic).
  function resetMasterGroup(groupName) {
    const group = groups.find((g) => g.name === groupName);
    if (!group) return;
    group.params.forEach((def) => {
      if (!applies(def, 'master')) return;
      if (def.def === undefined) return;
      sendAction(setValueAction(MASTER_COL, def, def.def));
    });
    if (groupName === 'VHS/NTSC') clearVhsPreset();
  }

  function randomizeMasterGroup(groupName) {
    const group = groups.find((g) => g.name === groupName);
    if (!group) return;
    group.params.forEach((def) => {
      if (!applies(def, 'master')) return;
      if (def.noRandom) return;
      if (def.ptype !== 'float' && def.ptype !== 'bipolar') return;
      const v = randInRange(def.min, def.max, def.step || 0.01);
      sendAction(setValueAction(MASTER_COL, def, v));
    });
    if (groupName === 'VHS/NTSC') flagVhsModifiedLocal();
  }

  // ---- VHS preset cycle + "modified" indicator ------------------------------
  function buildVhsPresetControl() {
    const wrap = document.createElement('span');
    wrap.className = 'mx-vhs-preset';
    wrap.title = 'Cycle VHS preset';
    const name = document.createElement('span');
    name.className = 'mx-vhs-preset-name';
    name.textContent = vhsPresetName || 'preset';
    wrap.appendChild(name);
    const mod = document.createElement('span');
    mod.className = 'mx-vhs-preset-mod';
    mod.textContent = '*';
    mod.style.display = 'none';
    wrap.appendChild(mod);
    vhsPresetEls = { name, mod };
    wrap.addEventListener('click', (e) => { e.stopPropagation(); cycleVhsPreset(1); });
    return wrap;
  }

  function presetNames() { return Object.keys(window.VHS_PRESETS || {}); }

  function cycleVhsPreset(dir) {
    const names = presetNames();
    if (!names.length) return;
    const i = names.indexOf(vhsPresetName);
    const next = i === -1 ? names[0] : names[(i + dir + names.length) % names.length];
    applyVhsPreset(next);
  }

  function applyVhsPreset(name) {
    const preset = (window.VHS_PRESETS || {})[name];
    if (!preset) return;
    // Direct sendAction batch (mirrors classic); flag stays clear until the user
    // tweaks a VHS control afterward.
    for (const [param, value] of Object.entries(preset)) {
      sendAction({ action: 'set_ntsc_param', param, value });
    }
    vhsPresetName = name;
    refreshVhsPreset(false);
  }

  function clearVhsPreset() {
    vhsPresetName = null;
    refreshVhsPreset(false);
  }

  function flagVhsModifiedLocal() {
    if (vhsPresetName) refreshVhsPreset(true);
  }

  function refreshVhsPreset(modified) {
    if (!vhsPresetEls) return;
    vhsPresetEls.name.textContent = vhsPresetName || 'preset';
    vhsPresetEls.mod.style.display = (vhsPresetName && modified) ? '' : 'none';
  }

  function updateMasterPanel(msg) {
    if (!masterBuilt) return;
    const [autos, errors] = readAutos(msg, MASTER_COL);
    masterRows.forEach((r) => updateCell(r.cell, msg, autos, errors));
    outputCells.forEach((c) => c.refresh());
    if (masterMeterFill) {
      const m = Math.max(0, Math.min(1, msg.meter || 0));
      masterMeterFill.style.width = (m * 100).toFixed(1) + '%';
    }
    syncVhsModified(msg);
  }

  // State-driven "modified" detection: once a preset is applied, mark it modified
  // when any VHS param in the live snapshot diverges from the preset's values.
  function syncVhsModified(msg) {
    if (!vhsPresetName || !vhsPresetEls) return;
    const preset = (window.VHS_PRESETS || {})[vhsPresetName];
    const ntsc = msg.ntsc || {};
    if (!preset) return;
    let modified = false;
    for (const [param, want] of Object.entries(preset)) {
      const got = ntsc[param];
      if (got === undefined) continue;
      if (typeof want === 'number') {
        if (Math.abs((got || 0) - want) > 1e-4) { modified = true; break; }
      } else if (got !== want) { modified = true; break; }
    }
    refreshVhsPreset(modified);
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
      val.textContent = 'OFF';
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
      const thumb = document.createElement('img');
      thumb.className = 'mx-clip-thumb';
      thumb.alt = '';
      el.appendChild(thumb);
      info.thumbEl = thumb;

      const val = document.createElement('span');
      val.className = 'mx-val mx-clip';
      el.appendChild(val);
      info.valEl = val;

      // Dice: swap to a random library clip without opening the modal.
      const dice = document.createElement('button');
      dice.className = 'mx-clip-rand';
      dice.type = 'button';
      dice.title = 'Random clip';
      dice.innerHTML = '<i data-lucide="dices"></i>';
      dice.addEventListener('click', (e) => {
        e.stopPropagation();
        randomClip(col.index);
      });
      el.appendChild(dice);

      // Click the clip cell to swap this layer's source video in place.
      el.classList.add('mx-clip-cell');
      el.title = 'Click to choose a different clip';
      el.addEventListener('click', () => openLibraryModal('swap', col.index));
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
  // Layer column reordering (drag the column header)
  // =====================================================================
  // Mirrors the classic view: pointer-driven so it survives the pointer leaving
  // the header. Rust owns order — we send move_layer and let the next snapshot
  // rebuild the grid in the new order. MASTER (kind !== 'layer') is never
  // draggable, so it stays pinned as the first column.
  let colDrag = null; // { fromIndex, headEl, target }

  function layerCols() {
    return columns.filter((c) => c.kind === 'layer' && c.headEl);
  }

  function clearColDropMarks() {
    gridEl.querySelectorAll('.mx-drop-before, .mx-drop-after')
      .forEach((e) => e.classList.remove('mx-drop-before', 'mx-drop-after'));
  }

  // Resolve which layer header the pointer is over (by X) and which edge.
  function colDropTarget(clientX) {
    clearColDropMarks();
    const heads = layerCols();
    if (!heads.length) return null;
    let best = null;
    for (const col of heads) {
      const r = col.headEl.getBoundingClientRect();
      if (clientX >= r.left && clientX <= r.right) {
        best = { col, before: clientX < r.left + r.width / 2 };
        break;
      }
    }
    if (!best) {
      const firstR = heads[0].headEl.getBoundingClientRect();
      best = clientX < firstR.left
        ? { col: heads[0], before: true }
        : { col: heads[heads.length - 1], before: false };
    }
    best.col.headEl.classList.add(best.before ? 'mx-drop-before' : 'mx-drop-after');
    return best;
  }

  function onColMove(e) {
    if (!colDrag) return;
    e.preventDefault();
    colDrag.target = colDropTarget(e.clientX);
  }

  function onColUp() {
    if (!colDrag) return;
    const from = colDrag.fromIndex;
    const t = colDrag.target;
    if (t) {
      const targetIndex = t.col.index;
      const insertion = t.before ? targetIndex : targetIndex + 1;
      // Account for the gap left by lifting the source out first.
      let to = from < insertion ? insertion - 1 : insertion;
      to = Math.max(0, Math.min(to, layerCols().length - 1));
      if (to !== from) sendAction({ action: 'move_layer', from, to });
    }
    if (colDrag.headEl) colDrag.headEl.classList.remove('mx-dragging');
    clearColDropMarks();
    colDrag = null;
    document.removeEventListener('pointermove', onColMove);
    document.removeEventListener('pointerup', onColUp);
  }

  function attachColumnDrag(headEl, col) {
    headEl.addEventListener('pointerdown', (e) => {
      if (e.button !== 0) return;
      e.preventDefault();
      colDrag = { fromIndex: col.index, headEl, target: null };
      headEl.classList.add('mx-dragging');
      document.addEventListener('pointermove', onColMove);
      document.addEventListener('pointerup', onColUp);
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
    updateMasterPanel(msg);
    syncClipProgress(msg);
    if (sidebarOpen) renderSidebar(msg);
  }

  // Clip-timing bar beneath each layer column header (L1, L2, …).
  function syncClipProgress(msg) {
    const layers = msg.layers || [];
    for (const col of columns) {
      if (col.kind !== 'layer' || !col.progFill) continue;
      const layer = layers[col.index];
      const p = layer ? Math.max(0, Math.min(1, layer.progress || 0)) : 0;
      col.progFill.style.width = (p * 100).toFixed(1) + '%';
    }
  }

  function ensureBuilt(msg) {
    if (!masterBuilt) buildMasterPanel();
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
      valEl.textContent = on ? 'ON' : 'OFF';
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
      if (info.thumbEl) {
        const want = v ? '/thumb/' + encodeURIComponent(v) : '';
        if (info.thumbEl.dataset.file !== (v || '')) {
          info.thumbEl.dataset.file = v || '';
          info.thumbEl.src = want;
          info.thumbEl.style.display = want ? '' : 'none';
        }
      }
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
    navRows = rows.filter((r) => !collapsed.has('layer:' + r.group));
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
    // Esc closes the library modal (handle before the input/focus guards so it
    // works even while the modal is the visual focus).
    const libModal = document.getElementById('library-modal');
    if (libModal && !libModal.hidden) {
      if (e.key === 'Escape') { closeLibraryModal(); e.preventDefault(); }
      return;
    }
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
  // Collapse state is scoped per grid ('layer:NAME' / 'master:NAME') so the
  // layer grid and the master panel expand/collapse the same-named group
  // independently.
  function toggleGroup(name, scope) {
    const key = scope + ':' + name;
    if (collapsed.has(key)) collapsed.delete(key);
    else collapsed.add(key);
    applyCollapse(name, scope, collapsed.has(key));
    if (scope === 'layer') {
      rebuildNav();
      ensureFocusable();
      paintFocus(false);
    }
  }

  function applyCollapse(name, scope, isCollapsed) {
    const root = scope === 'master' ? masterGridEl : gridEl;
    const sel = '[data-group="' + cssEscape(name) + '"]';
    root.querySelectorAll(sel).forEach((el) => {
      // The header band (group label + per-layer control cells) stays visible
      // when collapsed — only the param rows hide.
      if (el.classList.contains('mx-group') || el.classList.contains('mx-group-ctl')) {
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
    if (next === 'matrix' && lastMsg) {
      ensureBuilt(lastMsg);
      updateCells(lastMsg);
      updateMasterPanel(lastMsg);
      if (sidebarOpen) renderSidebar(lastMsg);
    }
  }

  let sidebarOpen = false;
  function toggleSidebar() {
    sidebarOpen = !sidebarOpen;
    app.classList.toggle('sidebar-open', sidebarOpen);
    if (sidebarOpen && lastMsg) renderSidebar(lastMsg);
  }

  let masterOpen = true;
  function toggleMaster() {
    masterOpen = !masterOpen;
    app.classList.toggle('master-open', masterOpen);
    if (masterOpen && lastMsg) updateMasterPanel(lastMsg);
  }

  function renderSidebar(msg) {
    renderPatches(msg.patches || []);
  }

  function renderPatches(patches) {
    const list = document.getElementById('mx-patches-list');
    const sig = patches.join('\u0001');
    if (list.dataset.sig === sig) return;
    list.dataset.sig = sig;
    list.innerHTML = '';

    if (!patches.length) {
      list.innerHTML = '<p class="dim" style="padding:6px 8px;">No saved patches</p>';
      return;
    }

    patches.forEach((name) => {
      const row = document.createElement('div');
      row.className = 'patch-row';

      const label = document.createElement('span');
      label.className = 'patch-name';
      label.textContent = name;
      label.title = 'Load "' + name + '"';
      label.addEventListener('click', () => sendAction({ action: 'load_patch', name }));
      row.appendChild(label);

      const del = document.createElement('button');
      del.className = 'patch-del';
      del.textContent = '\u00D7';
      del.title = 'Delete "' + name + '"';
      del.addEventListener('click', (e) => {
        e.stopPropagation();
        if (confirm('Delete patch "' + name + '"?')) sendAction({ action: 'delete_patch', name });
      });
      row.appendChild(del);

      list.appendChild(row);
    });
  }

  // The media library lives in a modal now. `libMode` is 'swap' (replace the
  // clip on libIndex's layer in place) or 'add' (append a new layer). The grid
  // is rebuilt each time the modal opens because the click target differs.
  let libMode = 'add';
  let libIndex = -1;

  function openLibraryModal(mode, index) {
    libMode = mode;
    libIndex = (typeof index === 'number') ? index : -1;
    const title = document.getElementById('library-modal-title');
    if (title) title.textContent = mode === 'swap' ? 'Swap clip' : 'Add layer';
    const grid = document.getElementById('library-modal-grid');
    if (grid) grid.dataset.sig = '';   // force re-render (handler depends on mode)
    renderLibrary((lastMsg && lastMsg.library) || []);
    document.getElementById('library-modal').hidden = false;
  }

  function closeLibraryModal() {
    document.getElementById('library-modal').hidden = true;
  }

  function renderLibrary(library) {
    const grid = document.getElementById('library-modal-grid');
    if (!grid) return;
    const sig = library.join('\u0001');
    if (grid.dataset.sig === sig) return;
    grid.dataset.sig = sig;
    grid.innerHTML = '';

    if (!library.length) {
      grid.innerHTML = '<p class="dim" style="grid-column:1/-1;text-align:center;padding:12px;">No media files</p>';
      return;
    }

    library.forEach((filename) => {
      const item = document.createElement('div');
      item.className = 'library-item';
      item.title = filename;

      // Server thumbnail; retry a few times while it is still being generated.
      const img = document.createElement('img');
      img.dataset.retries = '0';
      const thumbUrl = '/thumb/' + encodeURIComponent(filename);
      img.src = thumbUrl;
      img.onerror = () => {
        const retries = parseInt(img.dataset.retries, 10);
        if (retries < 5) {
          img.dataset.retries = String(retries + 1);
          setTimeout(() => { img.src = thumbUrl + '?r=' + (retries + 1); }, 1500 * (retries + 1));
        } else {
          img.style.display = 'none';
          const placeholder = document.createElement('span');
          placeholder.className = 'lib-placeholder';
          placeholder.textContent = filename.replace(/\.[^.]+$/, '');
          item.appendChild(placeholder);
        }
      };
      item.appendChild(img);

      // Hover preview animation (only cycles for clips that have /preview frames).
      let hoverInterval = null;
      let hovering = false;
      item.addEventListener('mouseenter', () => {
        hovering = true;
        const enc = encodeURIComponent(filename);
        const probe = new Image();
        probe.onload = () => {
          if (!hovering) return;
          let frame = 0;
          hoverInterval = setInterval(() => {
            frame = (frame + 1) % 8;
            img.src = '/preview/' + enc + '/' + frame;
          }, 250);
        };
        probe.src = '/preview/' + enc + '/0';
      });
      item.addEventListener('mouseleave', () => {
        hovering = false;
        if (hoverInterval) { clearInterval(hoverInterval); hoverInterval = null; }
        img.src = thumbUrl;
      });

      const label = document.createElement('span');
      label.className = 'lib-label';
      label.textContent = filename.replace(/\.[^.]+$/, '');
      item.appendChild(label);

      item.addEventListener('click', () => {
        if (libMode === 'swap' && libIndex >= 0) {
          sendAction({ action: 'set_layer_clip', index: libIndex, filename });
        } else {
          sendAction({ action: 'add_layer', filename });
        }
        closeLibraryModal();
      });
      grid.appendChild(item);
    });
  }

  // =====================================================================
  // Wiring
  // =====================================================================
  function init() {
    // Main (master) panel open by default; the right edge chevron collapses it.
    app.classList.add('master-open');

    // Default collapse: in the layer grid, every group except SOURCE starts
    // collapsed (seeded once here so user expand/collapse choices then persist
    // across rebuilds). Only `layer:` keys are seeded — the master panel is
    // untouched.
    layerLayout.forEach((g) => {
      if (g.name !== 'SOURCE') {
        collapsed.add('layer:' + g.name);
      }
    });

    const sb = document.getElementById('mx-sidebar-btn');
    if (sb) sb.addEventListener('click', toggleSidebar);
    const mb = document.getElementById('mx-master-btn');
    if (mb) mb.addEventListener('click', toggleMaster);
    const showWin = document.getElementById('mx-show-window-btn');
    if (showWin) showWin.addEventListener('click', () => sendAction({ action: 'focus_window' }));

    // Library modal dismissal: close button, backdrop click, Esc (in onKey).
    const libClose = document.getElementById('library-modal-close');
    if (libClose) libClose.addEventListener('click', closeLibraryModal);
    const libModal = document.getElementById('library-modal');
    if (libModal) libModal.addEventListener('click', (e) => {
      if (e.target === libModal) closeLibraryModal();
    });

    const saveBtn = document.getElementById('mx-patch-save');
    if (saveBtn) saveBtn.addEventListener('click', () => {
      const input = document.getElementById('mx-patch-name');
      const name = (input.value || '').trim();
      if (name) { sendAction({ action: 'save_patch', name }); input.value = ''; }
    });

    document.addEventListener('keydown', onKey);

    // Row hover: highlight the whole row (label + every cell) the pointer is over.
    // Wired independently on the layer grid and the master panel; each keeps its
    // own row index space and scopes its queries to its own root.
    attachRowHover(gridEl);
    attachRowHover(masterGridEl);

    // The whole trailing add column behaves as one big "add layer" button:
    // clicking anywhere in it opens the library in add mode, and hovering any
    // part of it lights up the entire column to advertise that.
    gridEl.addEventListener('click', (e) => {
      if (e.target.closest('.mx-addcol')) openLibraryModal('add');
    });
    gridEl.addEventListener('mouseover', (e) => {
      const hot = !!e.target.closest('.mx-addcol');
      addColCells.forEach((c) => c.classList.toggle('mx-addcol-hot', hot));
    });
    gridEl.addEventListener('mouseleave', () => {
      addColCells.forEach((c) => c.classList.remove('mx-addcol-hot'));
    });

    window.onMatrixState = syncMatrix;

    // Matrix is the default view. Boot into the legacy classic panel only when
    // explicitly requested via `?view=classic` (main.rs `--classic` flag).
    const bootView = new URLSearchParams(location.search).get('view');
    setView(bootView === 'classic' ? 'classic' : 'matrix');
  }

  // Per-root row-hover: each grid (layer grid + master panel) tracks its own
  // hovered row index and only toggles `.mx-row-hover` on its own descendants,
  // so the two grids' index spaces never collide.
  function attachRowHover(root) {
    if (!root) return;
    let hoverRow = null;
    const setHoverRow = (idx) => {
      if (hoverRow === idx) return;
      if (hoverRow !== null) {
        root.querySelectorAll('[data-mxrow="' + hoverRow + '"]')
          .forEach((e) => e.classList.remove('mx-row-hover'));
      }
      hoverRow = idx;
      if (hoverRow !== null) {
        root.querySelectorAll('[data-mxrow="' + hoverRow + '"]')
          .forEach((e) => e.classList.add('mx-row-hover'));
      }
    };
    root.addEventListener('mouseover', (e) => {
      const el = e.target.closest('[data-mxrow]');
      setHoverRow(el ? el.dataset.mxrow : null);
    });
    root.addEventListener('mouseleave', () => setHoverRow(null));
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();

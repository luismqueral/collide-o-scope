// collide-o-scope — web control panel

const statusEl = document.getElementById('ws-status');
const layersList = document.getElementById('layers-list');
const layersEmpty = document.getElementById('layers-empty');
const libraryGrid = document.getElementById('library-grid');
const patchesList = document.getElementById('patches-list');

// --- WebSocket ---

let ws;
function connect() {
  ws = new WebSocket(`ws://${location.host}/ws`);

  ws.onopen = () => {
    statusEl.classList.add('connected');
    statusEl.classList.remove('disconnected');
    statusEl.title = 'connected';
  };

  ws.onclose = () => {
    statusEl.classList.remove('connected');
    statusEl.classList.add('disconnected');
    statusEl.title = 'disconnected';
    setTimeout(connect, 2000);
  };

  ws.onmessage = (e) => {
    if (e.data instanceof ArrayBuffer) return;

    try {
      const msg = JSON.parse(e.data);
      if (msg.type === 'state') {
        syncEffects(msg.effects, msg.automations, msg.automation_errors);
        syncNtsc(msg.ntsc);
        syncFramerate(msg.framerate);
        syncOutput(msg);
        syncLayers(msg.layers);
        syncLibrary(msg.library);
        syncPatches(msg.patches || []);
        syncTransport(msg.paused);
        syncExport(msg.export_progress, msg.export_error);
        syncTempo(msg.bpm, msg.beat);
      }
    } catch (err) {
      console.warn('[ws] parse error:', err);
    }
  };
}
connect();

function sendAction(action) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    console.log('[ws] send:', JSON.stringify(action));
    ws.send(JSON.stringify(action));
  } else {
    console.warn('[ws] not connected, dropping:', action);
  }
}

// --- Parameter automation: click-to-edit value cells ---
//
// Clicking a numeric param's value turns it into a text field. Typing a plain
// number sends a normal set (and clears any existing automation); typing a
// formula like `sin(t)*10` installs an automation that the Rust render loop
// evaluates every frame. The `ƒ` marker and error styling are driven by the
// `automations` / `automation_errors` maps in the server snapshot.

// A value matching this is treated as a plain number, not a formula.
const NUMERIC_RE = /^-?\d*\.?\d+$/;

// Wire a `.value` span so it becomes editable on click. `ctx` provides the
// actions to send: setValue(num), setAutomation(expr), clearAutomation().
function makeValueEditable(valueEl, ctx) {
  if (!valueEl || valueEl.dataset.editable === '1') return;
  valueEl.dataset.editable = '1';

  valueEl.addEventListener('click', () => {
    if (valueEl.querySelector('input')) return; // already editing
    const current = valueEl.dataset.expr || valueEl.textContent.trim();
    const input = document.createElement('input');
    input.type = 'text';
    input.className = 'value-input';
    input.value = current;
    valueEl.textContent = '';
    valueEl.appendChild(input);
    input.focus();
    input.select();

    let committed = false;
    const commit = () => {
      if (committed) return;
      committed = true;
      const text = input.value.trim();
      if (text === '') {
        // Emptying the field removes an active formula (the intuitive "delete"
        // gesture). With no formula installed it's just a no-op.
        if (valueEl.dataset.expr) ctx.clearAutomation();
      } else if (NUMERIC_RE.test(text)) {
        ctx.clearAutomation();
        ctx.setValue(parseFloat(text));
      } else {
        ctx.setAutomation(text);
      }
      input.blur();
    };
    const cancel = () => {
      committed = true;
      input.blur();
    };

    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') { e.preventDefault(); commit(); }
      else if (e.key === 'Escape') { e.preventDefault(); cancel(); }
    });
    input.addEventListener('blur', () => {
      if (!committed) commit();
      // Repaint from the latest known state until the next server sync arrives.
      input.remove();
    });
  });
}

// Apply automation state from a snapshot map onto a value cell: toggle the
// `automated` class + remember the source (shown when editing), and the
// `error` class + tooltip when the formula failed to parse.
function applyAutomationState(valueEl, param, autos, errors) {
  if (!valueEl) return;
  const expr = autos && autos[param];
  const err = errors && errors[param];
  if (expr) {
    valueEl.classList.add('automated');
    valueEl.dataset.expr = expr;
  } else {
    valueEl.classList.remove('automated');
    delete valueEl.dataset.expr;
  }
  if (err) {
    valueEl.classList.add('error');
    valueEl.title = err;
  } else {
    valueEl.classList.remove('error');
    valueEl.removeAttribute('title');
  }
  // The per-param ƒ launcher doubles as the "this is animated" indicator.
  const launcher = valueEl.closest('.param-row')?.querySelector('.fx-launcher');
  if (launcher) launcher.classList.toggle('active', !!expr);
}

// --- Initialize sliders from DOM attributes ---

document.querySelectorAll('.param-row[data-param]').forEach((row) => {
  const param = row.dataset.param;
  const min = parseFloat(row.dataset.min);
  const max = parseFloat(row.dataset.max);
  const step = parseFloat(row.dataset.step);

  const slider = row.querySelector('input[type="range"]');
  const valueEl = row.querySelector('.value');
  const checkbox = row.querySelector('input[type="checkbox"]');
  const select = row.querySelector('select');

  if (slider) {
    slider.min = min;
    slider.max = max;
    slider.step = step;
    slider.value = min;

    slider.addEventListener('input', () => {
      const v = parseFloat(slider.value);
      valueEl.textContent = formatValue(v, min, max, step);
      sendAction({ action: 'set_param', param, value: v });
    });

    // Numeric master params can also be automated by clicking the value.
    makeValueEditable(valueEl, {
      setValue: (v) => sendAction({ action: 'set_param', param, value: v }),
      setAutomation: (expr) => sendAction({ action: 'set_automation', param, expr }),
      clearAutomation: () => sendAction({ action: 'clear_automation', param }),
    });

    // …and via the guided modulator builder (the ƒ launcher).
    addLauncher(row, {
      scope: 'master', param, lo: min, hi: max, step,
      label: (row.querySelector('label')?.textContent || param).trim(),
    });
  }

  if (checkbox) {
    checkbox.addEventListener('change', () => {
      sendAction({ action: 'set_param', param, value: checkbox.checked });
    });
  }

  if (select) {
    select.addEventListener('change', () => {
      sendAction({ action: 'set_param', param, value: parseInt(select.value) });
    });
  }
});

// --- Master content framerate (frame-hold / stutter) ---
// Distinct from set_param effects: routed via its own action so the generic
// loop above (which sends set_param) doesn't pick it up.
const fpsRow = document.querySelector('.param-row[data-master-param="framerate"]');
if (fpsRow) {
  const min = parseFloat(fpsRow.dataset.min);
  const max = parseFloat(fpsRow.dataset.max);
  const step = parseFloat(fpsRow.dataset.step);
  const slider = fpsRow.querySelector('input[type="range"]');
  const valueEl = fpsRow.querySelector('.value');
  slider.min = min;
  slider.max = max;
  slider.step = step;
  slider.value = max; // default 30 = smooth
  valueEl.textContent = formatValue(max, min, max, step);
  slider.addEventListener('input', () => {
    const v = parseFloat(slider.value);
    valueEl.textContent = formatValue(v, min, max, step);
    sendAction({ action: 'set_master_framerate', value: v });
  });
}

// --- Master output size / aspect ratio ---
// Two selects routed via their own action (set_output_size), not the generic
// data-param loop. Latest synced dims feed the export modal (one source of truth).
let lastOutputW = 1920, lastOutputH = 1080;
const ratioSel = document.querySelector('[data-master-param="output_ratio"] select');
const qualSel = document.querySelector('[data-master-param="output_quality"] select');
const sendOutput = () => sendAction({
  action: 'set_output_size',
  ratio: ratioSel.value,
  quality: parseInt(qualSel.value, 10),
});
ratioSel?.addEventListener('change', sendOutput);
qualSel?.addEventListener('change', sendOutput);

// Idempotent sync from server. Skip a control while it's focused so we don't
// fight the user mid-selection. Stores dims for the export modal.
function syncOutput(msg) {
  if (!msg) return;
  if (ratioSel && document.activeElement !== ratioSel && msg.output_ratio != null) {
    ratioSel.value = msg.output_ratio;
  }
  if (qualSel && document.activeElement !== qualSel && msg.output_quality != null) {
    qualSel.value = String(msg.output_quality);
  }
  if (msg.output_width) lastOutputW = msg.output_width;
  if (msg.output_height) lastOutputH = msg.output_height;
  const d = document.getElementById('export-resolution-display');
  if (d && msg.output_width && msg.output_height) {
    d.textContent = `${msg.output_width}×${msg.output_height}`;
  }
}

// --- Initialize NTSC/VHS sliders ---

document.querySelectorAll('.param-row[data-ntsc]').forEach((row) => {
  const param = row.dataset.ntsc;
  const min = parseFloat(row.dataset.min);
  const max = parseFloat(row.dataset.max);
  const step = parseFloat(row.dataset.step);

  const slider = row.querySelector('input[type="range"]');
  const valueEl = row.querySelector('.value');
  const checkbox = row.querySelector('input[type="checkbox"]');
  const select = row.querySelector('select');

  if (slider) {
    slider.min = min;
    slider.max = max;
    slider.step = step;
    slider.value = min;

    slider.addEventListener('input', () => {
      const v = parseFloat(slider.value);
      valueEl.textContent = formatValue(v, min, max, step);
      sendAction({ action: 'set_ntsc_param', param, value: v });
      flagVhsModified();
    });
  }

  if (checkbox) {
    checkbox.addEventListener('change', () => {
      sendAction({ action: 'set_ntsc_param', param, value: checkbox.checked });
      flagVhsModified();
    });
  }

  if (select) {
    select.addEventListener('change', () => {
      sendAction({ action: 'set_ntsc_param', param, value: parseInt(select.value) });
      flagVhsModified();
    });
  }
});

// --- Collapsible FX groups ---

document.querySelectorAll('.fx-group-header').forEach((header) => {
  header.addEventListener('click', (e) => {
    // Don't collapse when interacting with header controls (reset/randomize/preset).
    if (e.target.closest('.group-reset, .group-rand, .vhs-preset, .preset-modified')) return;
    const group = header.closest('.fx-group');
    group.classList.toggle('collapsed');
  });
});

// --- Group reset buttons ---
//
// Reset returns params to their defaults — and should also remove any formula
// driving them. Otherwise the automation would just re-write the value on the
// next frame. We clear the formulas FIRST, then reset the values, so nothing
// over-writes the param after it has been reset back to default.

function clearFormulasIn(root) {
  if (!root) return;
  // `.automated` = a live formula; `.error` = a formula that failed to parse.
  // Both should go on reset; clearing a param with no formula is a no-op.
  root.querySelectorAll('.value.automated, .value.error').forEach((valueEl) => {
    const param = valueEl.closest('.param-row')?.dataset.param;
    if (param) sendAction({ action: 'clear_automation', param });
  });
}

document.querySelectorAll('.group-reset').forEach((btn) => {
  btn.addEventListener('click', (e) => {
    e.stopPropagation();
    clearFormulasIn(btn.closest('.fx-group'));
    sendAction({ action: 'reset_group', group: btn.dataset.group });
  });
});

// --- Master group randomize (sliders only) ---

document.querySelectorAll('.group-rand').forEach((btn) => {
  btn.addEventListener('click', (e) => {
    e.stopPropagation();
    randomizeGroup(btn.closest('.fx-group'));
  });
});

// Randomize every range slider in a master FX group, leaving toggles/selects alone.
function randomizeGroup(group) {
  if (!group) return;
  const body = group.querySelector('.fx-group-body');
  if (!body) return;
  body.querySelectorAll('.param-row').forEach((row) => {
    if (row.dataset.masterParam) return; // skip master framerate (own action, not an effect)
    const slider = row.querySelector('input[type="range"]');
    if (!slider) return; // sliders only — skip toggles/selects
    const min = parseFloat(slider.min);
    const max = parseFloat(slider.max);
    const step = parseFloat(slider.step) || 0.01;
    const v = randInRange(min, max, step);
    slider.value = v;
    const valEl = row.querySelector('.value');
    if (valEl) valEl.textContent = formatValue(v, min, max, step);
    if (row.dataset.param) {
      sendAction({ action: 'set_param', param: row.dataset.param, value: v });
    } else if (row.dataset.ntsc) {
      sendAction({ action: 'set_ntsc_param', param: row.dataset.ntsc, value: v });
    }
  });
  if (group.id === 'vhs-group') flagVhsModified();
}

// --- VHS presets ---
// Full parameter sets keyed by `data-ntsc` field names. Applying a preset sends
// one set_ntsc_param per field; the next state push syncs the DOM controls.

const VHS_PRESETS = {
  'Clean': {
    enabled: true, tape_speed: 0,
    head_switching_enabled: false, head_switching_height: 8, head_switching_shift: 0,
    tracking_noise_enabled: false, tracking_noise_height: 24, tracking_noise_wave: 0, tracking_noise_snow: 0,
    snow_intensity: 0, composite_noise_intensity: 0, luma_noise_intensity: 0, chroma_noise_intensity: 0, chroma_loss: 0,
    edge_wave_enabled: false, edge_wave_intensity: 0, edge_wave_speed: 0.5,
    luma_smear: 0, composite_sharpening: 0.5,
  },
  'Worn Tape': {
    enabled: true, tape_speed: 1,
    head_switching_enabled: false, head_switching_height: 8, head_switching_shift: 0,
    tracking_noise_enabled: false, tracking_noise_height: 24, tracking_noise_wave: 0, tracking_noise_snow: 0,
    snow_intensity: 0.05, composite_noise_intensity: 0.08, luma_noise_intensity: 0.04, chroma_noise_intensity: 0.10, chroma_loss: 0.002,
    edge_wave_enabled: true, edge_wave_intensity: 3, edge_wave_speed: 1.0,
    luma_smear: 0.15, composite_sharpening: 0.8,
  },
  'Heavy Damage': {
    enabled: true, tape_speed: 2,
    head_switching_enabled: true, head_switching_height: 12, head_switching_shift: 40,
    tracking_noise_enabled: true, tracking_noise_height: 60, tracking_noise_wave: 20, tracking_noise_snow: 0.5,
    snow_intensity: 0.30, composite_noise_intensity: 0.25, luma_noise_intensity: 0.12, chroma_noise_intensity: 0.30, chroma_loss: 0.006,
    edge_wave_enabled: true, edge_wave_intensity: 10, edge_wave_speed: 2.0,
    luma_smear: 0.4, composite_sharpening: 1.2,
  },
  'Tracking Trouble': {
    enabled: true, tape_speed: 1,
    head_switching_enabled: true, head_switching_height: 16, head_switching_shift: 60,
    tracking_noise_enabled: true, tracking_noise_height: 90, tracking_noise_wave: 35, tracking_noise_snow: 0.7,
    snow_intensity: 0.2, composite_noise_intensity: 0.10, luma_noise_intensity: 0, chroma_noise_intensity: 0, chroma_loss: 0,
    edge_wave_enabled: true, edge_wave_intensity: 6, edge_wave_speed: 3.0,
    luma_smear: 0.2, composite_sharpening: 0,
  },
};

let currentVhsPreset = null;
let vhsModified = false;

const vhsPresetSelect = document.querySelector('.vhs-preset');
const presetModifiedEl = document.querySelector('.preset-modified');

if (vhsPresetSelect) {
  vhsPresetSelect.addEventListener('change', (e) => {
    e.stopPropagation();
    const name = vhsPresetSelect.value;
    const preset = VHS_PRESETS[name];
    if (!preset) return;
    // Direct sendAction (no DOM events dispatched) so this never self-flags as modified.
    for (const [param, value] of Object.entries(preset)) {
      sendAction({ action: 'set_ntsc_param', param, value });
    }
    currentVhsPreset = name;
    vhsModified = false;
    if (presetModifiedEl) presetModifiedEl.style.display = 'none';
  });
}

// Mark the active preset as modified once the user tweaks any VHS control.
function flagVhsModified() {
  if (!currentVhsPreset || vhsModified) return;
  vhsModified = true;
  if (presetModifiedEl) presetModifiedEl.style.display = '';
}

// --- Transport buttons ---

document.getElementById('btn-play-all').addEventListener('click', () => {
  sendAction({ action: 'toggle_master_pause' });
});

// Reset FX resets every master param, so clear every master formula too.
document.getElementById('btn-stop').addEventListener('click', () => {
  clearFormulasIn(document.getElementById('master-fx'));
  sendAction({ action: 'reset_fx' });
});

// --- Tap tempo + BPM readout ---
//
// The browser only sends a bare `tap_tempo`; the Rust render loop timestamps it
// against the same elapsed clock the formulas use, so the musical `beat` phase
// lines up with the tap downbeat. Manual entry sends `set_bpm` (no phase reset).

function tapTempo() {
  sendAction({ action: 'tap_tempo' });
}

// Tap tempo is a helper *inside* the automation editor (for building beat-synced
// formulas), not a global transport control. The button + readout live in the
// modal; the bpm/beat STATE stays global because every formula references
// `beat`/`bpm` in the render loop.
document.getElementById('ae-tap').addEventListener('click', tapTempo);

// `T` taps tempo only while the editor is open — ignore it elsewhere / typing.
document.addEventListener('keydown', (e) => {
  if (e.key !== 't' && e.key !== 'T') return;
  if (!isEditorOpen()) return;
  const tag = (e.target.tagName || '').toLowerCase();
  if (tag === 'input' || tag === 'textarea' || tag === 'select') return;
  tapTempo();
});

// Click the editor's BPM readout to type an exact tempo. Reuses the value-cell
// editor; BPM can't be automated, so the automation hooks are no-ops.
makeValueEditable(document.getElementById('ae-bpm'), {
  setValue: (v) => sendAction({ action: 'set_bpm', value: v }),
  setAutomation: () => {},
  clearAutomation: () => {},
});

// Pulse the beat dot once per beat. Track the last whole-beat index so we only
// toggle on a change (no per-frame DOM churn).
let lastBeatIndex = -1;
function syncTempo(bpm, beat) {
  if (typeof bpm !== 'number') return;
  // Feed the editor preview so beat-synced plots ride the live tempo.
  liveBpm = bpm > 0 ? bpm : 120;
  if (typeof beat === 'number') liveBeat = beat;
  const readout = document.getElementById('ae-bpm');
  // Don't clobber the field while the user is typing a tempo.
  if (readout && !readout.querySelector('input')) {
    readout.textContent = String(Math.round(bpm));
  }
  // The dot only shows inside the editor, so skip the pulse work when closed.
  const dot = document.getElementById('ae-beat-dot');
  if (dot && isEditorOpen() && typeof beat === 'number') {
    const idx = Math.floor(beat);
    if (idx !== lastBeatIndex) {
      lastBeatIndex = idx;
      dot.classList.add('on');
      // Remove on the next frame so the CSS transition can re-trigger.
      requestAnimationFrame(() => requestAnimationFrame(() => dot.classList.remove('on')));
    }
  }
}

// --- Sync effects UI from server ---

function syncEffects(effects, automations, automationErrors) {
  if (!effects) return;
  for (const [param, value] of Object.entries(effects)) {
    const row = document.querySelector(`.param-row[data-param="${param}"]`);
    if (!row) continue;

    const slider = row.querySelector('input[type="range"]');
    const valueEl = row.querySelector('.value');
    const checkbox = row.querySelector('input[type="checkbox"]');
    const select = row.querySelector('select');

    if (slider && valueEl && document.activeElement !== slider) {
      slider.value = value;
      // While editing the value cell, don't overwrite the user's text.
      if (!valueEl.querySelector('input')) {
        const min = parseFloat(row.dataset.min);
        const max = parseFloat(row.dataset.max);
        const step = parseFloat(row.dataset.step);
        valueEl.textContent = formatValue(value, min, max, step);
      }
      applyAutomationState(valueEl, param, automations, automationErrors);
    }

    if (checkbox) {
      checkbox.checked = !!value;
    }

    if (select) {
      select.value = value;
    }
  }
}

// --- Sync master framerate from server ---

function syncFramerate(framerate) {
  if (framerate == null) return;
  const row = document.querySelector('.param-row[data-master-param="framerate"]');
  if (!row) return;
  const slider = row.querySelector('input[type="range"]');
  const valueEl = row.querySelector('.value');
  if (slider && valueEl && document.activeElement !== slider) {
    slider.value = framerate;
    const min = parseFloat(row.dataset.min);
    const max = parseFloat(row.dataset.max);
    const step = parseFloat(row.dataset.step);
    valueEl.textContent = formatValue(framerate, min, max, step);
  }
}

// --- Sync NTSC/VHS UI from server ---

function syncNtsc(ntsc) {
  if (!ntsc) return;
  for (const [param, value] of Object.entries(ntsc)) {
    const row = document.querySelector(`.param-row[data-ntsc="${param}"]`);
    if (!row) continue;

    const slider = row.querySelector('input[type="range"]');
    const valueEl = row.querySelector('.value');
    const checkbox = row.querySelector('input[type="checkbox"]');
    const select = row.querySelector('select');

    if (slider && valueEl && document.activeElement !== slider) {
      slider.value = value;
      const min = parseFloat(row.dataset.min);
      const max = parseFloat(row.dataset.max);
      const step = parseFloat(row.dataset.step);
      valueEl.textContent = formatValue(value, min, max, step);
    }

    if (checkbox) {
      checkbox.checked = !!value;
    }

    if (select && document.activeElement !== select) {
      select.value = value;
    }
  }
}

// --- Layer event delegation ---
// All layer-card controls are handled here (not per-card) so that indices are
// read live from `card.dataset.index` and stay correct across reordering.

layersList.addEventListener('click', (e) => {
  const card = e.target.closest('.layer-card');
  if (!card) return;
  const index = parseInt(card.dataset.index);

  if (e.target.closest('.layer-fx-rand')) {
    e.stopPropagation();
    randomizeLayerGroup(e.target.closest('.fx-group'), index);
  } else if (e.target.closest('.layer-thumb-wrap')) {
    e.stopPropagation();
    sendAction({ action: 'toggle_layer_pause', index });
  } else if (e.target.closest('.layer-vis-btn')) {
    e.stopPropagation();
    sendAction({ action: 'toggle_visibility', index });
  } else if (e.target.closest('.layer-remove-btn')) {
    e.stopPropagation();
    sendAction({ action: 'remove_layer', index });
  } else if (e.target.closest('.fx-group-header')) {
    // Collapse/expand a single FX group inside the layer body.
    e.target.closest('.fx-group').classList.toggle('collapsed');
  } else if (e.target.closest('.layer-grip')) {
    // grip is for dragging only — don't toggle expand
  } else if (e.target.closest('.layer-header')) {
    card.classList.toggle('expanded');
  }
});

// Double-click a layer's title bar to collapse/expand all its FX groups at once.
// Excludes the interactive header controls so they keep their own behavior.
layersList.addEventListener('dblclick', (e) => {
  const header = e.target.closest('.layer-header');
  if (!header) return;
  if (e.target.closest('.layer-thumb-wrap, .layer-vis-btn, .layer-remove-btn, .layer-grip')) return;
  const card = header.closest('.layer-card');
  const groups = card.querySelectorAll('.layer-body .fx-group');
  const anyOpen = Array.from(groups).some((g) => !g.classList.contains('collapsed'));
  groups.forEach((g) => g.classList.toggle('collapsed', anyOpen));
});

layersList.addEventListener('input', (e) => {
  const slider = e.target;
  if (slider.type !== 'range') return;
  const row = slider.closest('.param-row[data-param]');
  const card = slider.closest('.layer-card');
  if (!row || !card) return;
  const index = parseInt(card.dataset.index);
  const param = row.dataset.param;
  const v = parseFloat(slider.value);
  const valEl = row.querySelector('.value');
  if (valEl) valEl.textContent = formatValue(v, parseFloat(slider.min), parseFloat(slider.max), parseFloat(slider.step));
  sendAction({ action: 'set_layer_param', index, param, value: v });
});

layersList.addEventListener('change', (e) => {
  const el = e.target;
  const row = el.closest('.param-row[data-param]');
  const card = el.closest('.layer-card');
  if (!row || !card) return;
  const index = parseInt(card.dataset.index);
  const param = row.dataset.param;
  if (el.tagName === 'SELECT') {
    sendAction({ action: 'set_layer_param', index, param, value: el.value });
  } else if (el.type === 'checkbox') {
    sendAction({ action: 'set_layer_param', index, param, value: el.checked });
  } else if (el.type === 'color') {
    sendAction({ action: 'set_layer_param', index, param, value: el.value });
  }
});

// --- Layer reorder (pointer events) ---
// HTML5 drag-and-drop proved flaky on non-first cards, so we drive reorder with
// raw pointer events instead. Pointer capture isn't used; we hang move/up
// listeners on `document` so the gesture survives the pointer leaving the grip.
// Rust is authoritative for order: we send move_layer and let the next state
// push re-render. `dragging` blocks syncLayers from churning mid-drag.

let dragSrcIndex = null;
let dragSrcCard = null;
let dragging = false;

function clearDropMarkers() {
  layersList.querySelectorAll('.drop-before, .drop-after').forEach((c) => {
    c.classList.remove('drop-before', 'drop-after');
  });
}

// Locate the card under the pointer (elementFromPoint sees the topmost node, so
// walk up to its .layer-card) and mark which edge we'd drop against.
function updateDropTarget(clientY) {
  clearDropMarkers();
  const el = document.elementFromPoint(lastPointerX, clientY);
  const card = el && el.closest('.layer-card');
  if (!card || !layersList.contains(card)) return;
  const rect = card.getBoundingClientRect();
  const before = (clientY - rect.top) < rect.height / 2;
  card.classList.add(before ? 'drop-before' : 'drop-after');
}

let lastPointerX = 0;

function onDragMove(e) {
  if (!dragging) return;
  e.preventDefault();
  lastPointerX = e.clientX;
  updateDropTarget(e.clientY);
}

function onDragEnd(e) {
  if (!dragging) return;
  // Resolve the drop position from whichever card carries a marker.
  const beforeCard = layersList.querySelector('.drop-before');
  const afterCard = layersList.querySelector('.drop-after');
  const marked = beforeCard || afterCard;
  if (marked) {
    const targetIndex = parseInt(marked.dataset.index);
    const insertion = beforeCard ? targetIndex : targetIndex + 1;
    // Adjust for the gap left by removing the source first.
    let to = dragSrcIndex < insertion ? insertion - 1 : insertion;
    to = Math.max(0, Math.min(to, layersList.children.length - 1));
    if (to !== dragSrcIndex) {
      sendAction({ action: 'move_layer', from: dragSrcIndex, to });
    }
  }
  if (dragSrcCard) dragSrcCard.classList.remove('dragging');
  layersList.classList.remove('reordering');
  clearDropMarkers();
  dragging = false;
  dragSrcIndex = null;
  dragSrcCard = null;
  document.removeEventListener('pointermove', onDragMove);
  document.removeEventListener('pointerup', onDragEnd);
}

layersList.addEventListener('pointerdown', (e) => {
  const grip = e.target.closest('.layer-grip');
  if (!grip) return;
  e.preventDefault();
  const card = grip.closest('.layer-card');
  if (!card) return;
  dragSrcIndex = parseInt(card.dataset.index);
  dragSrcCard = card;
  dragging = true;
  lastPointerX = e.clientX;
  card.classList.add('dragging');
  layersList.classList.add('reordering'); // CSS collapses all bodies while dragging
  document.addEventListener('pointermove', onDragMove);
  document.addEventListener('pointerup', onDragEnd);
});

// --- Sync layers ---

function syncLayers(layers) {
  if (!layers) return;
  if (dragging) return; // don't reorder DOM mid-drag
  layersEmpty.style.display = layers.length === 0 ? 'block' : 'none';

  // Reconcile by stable layer id. Re-appending an existing node *relocates* it,
  // so the whole card (thumbnail, expanded state, slider DOM) follows its layer
  // across reorders instead of being morphed in place.
  const existing = new Map();
  Array.from(layersList.children).forEach((card) => {
    existing.set(card.dataset.id, card);
  });

  // Drop cards whose layer no longer exists.
  const liveIds = new Set(layers.map((l) => String(l.id)));
  existing.forEach((card, id) => {
    if (!liveIds.has(id)) {
      card.remove();
      existing.delete(id);
    }
  });

  // Place cards in server order. CRITICAL: only touch the DOM when a card is
  // actually out of position. An unconditional appendChild here detaches and
  // reattaches every card on every state push (~30fps), which rips nodes out
  // from under the pointer mid-click/drag (broken buttons, broken reorder) and
  // makes the list flicker. insertBefore is a no-op-free reorder: untouched
  // when order is unchanged, minimal moves when it isn't.
  layers.forEach((layer, i) => {
    const id = String(layer.id);
    let card = existing.get(id);
    if (!card) card = createLayerCard(layer, i);
    const atPos = layersList.children[i];
    if (atPos !== card) {
      layersList.insertBefore(card, atPos || null);
    }
    updateLayerCard(card, layer, i);
  });
}

function createLayerCard(layer, index) {
  const card = document.createElement('div');
  card.className = 'layer-card expanded';
  card.dataset.index = index;
  card.dataset.id = layer.id;

  card.innerHTML = `
    <div class="layer-header">
      <span class="layer-grip" title="Drag to reorder"><i data-lucide="grip-vertical"></i></span>
      <span class="layer-thumb-wrap ${layer.paused ? 'paused' : ''}" title="Play/Pause">
        <img class="layer-thumb" src="/thumb/${encodeURIComponent(layer.filename)}" alt="">
        <span class="layer-thumb-state"><i data-lucide="play"></i></span>
      </span>
      <span class="layer-num">${index + 1}</span>
      <span class="layer-title">${layer.filename || 'Untitled'}</span>
      <button class="layer-vis-btn ${layer.visible ? 'visible' : ''}" data-visible="${layer.visible}" title="Visibility"><i data-lucide="${layer.visible ? 'eye' : 'eye-off'}"></i></button>
      <button class="layer-remove-btn" title="Remove"><i data-lucide="x"></i></button>
    </div>
    <div class="layer-progress"><div class="layer-progress-fill" style="width:${(layer.progress * 100).toFixed(1)}%"></div></div>
    <div class="layer-body">
      <div class="fx-group" data-layer-group="blend">
        <div class="fx-group-header">
          <span class="chevron">&#x25BC;</span>
          <span class="group-label">BLEND</span>
        </div>
        <div class="fx-group-body">
          <div class="param-row" data-param="opacity">
            <label>Opacity</label>
            <input type="range" min="0" max="1" step="0.01" value="${layer.opacity}">
            <span class="value">${layer.opacity.toFixed(2)}</span>
          </div>
          <div class="param-row" data-param="speed">
            <label>Speed</label>
            <input type="range" min="0.25" max="4" step="0.25" value="${layer.speed}">
            <span class="value">${layer.speed.toFixed(2)}</span>
          </div>
          <div class="param-row" data-param="fps" data-min="1" data-max="30" data-step="1">
            <label>FPS</label>
            <input type="range" min="1" max="30" step="1" value="${layer.fps}">
            <span class="value">${layer.fps.toFixed(0)}</span>
          </div>
          <div class="param-row select-row" data-param="blend_mode">
            <label>Blend</label>
            <select>
              <option value="normal" ${layer.blend_mode === 'normal' ? 'selected' : ''}>Normal</option>
              <option value="screen" ${layer.blend_mode === 'screen' ? 'selected' : ''}>Screen</option>
              <option value="multiply" ${layer.blend_mode === 'multiply' ? 'selected' : ''}>Multiply</option>
              <option value="difference" ${layer.blend_mode === 'difference' ? 'selected' : ''}>Difference</option>
            </select>
          </div>
        </div>
      </div>

      <div class="fx-group collapsed" data-layer-group="color">
        <div class="fx-group-header">
          <span class="chevron">&#x25BC;</span>
          <span class="group-label">COLOR</span>
          <button class="layer-fx-rand" title="Randomize"><i data-lucide="dices"></i></button>
        </div>
        <div class="fx-group-body">
          <div class="param-row" data-param="hue_shift">
            <label>Hue</label>
            <input type="range" min="-180" max="180" step="1" value="${layer.hue_shift}">
            <span class="value">${formatValue(layer.hue_shift, -180, 180, 1)}</span>
          </div>
          <div class="param-row" data-param="saturation">
            <label>Saturation</label>
            <input type="range" min="-1" max="1" step="0.01" value="${layer.saturation}">
            <span class="value">${formatValue(layer.saturation, -1, 1, 0.01)}</span>
          </div>
          <div class="param-row" data-param="brightness">
            <label>Brightness</label>
            <input type="range" min="-1" max="1" step="0.01" value="${layer.brightness}">
            <span class="value">${formatValue(layer.brightness, -1, 1, 0.01)}</span>
          </div>
          <div class="param-row" data-param="contrast">
            <label>Contrast</label>
            <input type="range" min="-1" max="1" step="0.01" value="${layer.contrast}">
            <span class="value">${formatValue(layer.contrast, -1, 1, 0.01)}</span>
          </div>
        </div>
      </div>

      <div class="fx-group collapsed" data-layer-group="digital">
        <div class="fx-group-header">
          <span class="chevron">&#x25BC;</span>
          <span class="group-label">DIGITAL</span>
          <button class="layer-fx-rand" title="Randomize"><i data-lucide="dices"></i></button>
        </div>
        <div class="fx-group-body">
          <div class="param-row" data-param="pixelate">
            <label>Pixelate</label>
            <input type="range" min="1" max="32" step="1" value="${layer.pixelate}">
            <span class="value">${formatValue(layer.pixelate, 1, 32, 1)}</span>
          </div>
          <div class="param-row" data-param="rgb_split">
            <label>RGB Split</label>
            <input type="range" min="0" max="30" step="0.5" value="${layer.rgb_split}">
            <span class="value">${formatValue(layer.rgb_split, 0, 30, 0.5)}</span>
          </div>
          <div class="param-row" data-param="posterize">
            <label>Posterize</label>
            <input type="range" min="0" max="16" step="1" value="${layer.posterize}">
            <span class="value">${formatValue(layer.posterize, 0, 16, 1)}</span>
          </div>
          <div class="param-row toggle-row" data-param="invert">
            <label>Invert</label>
            <label class="toggle"><input type="checkbox" ${layer.invert ? 'checked' : ''}><span class="toggle-slider"></span></label>
          </div>
        </div>
      </div>

      <div class="fx-group collapsed" data-layer-group="warp">
        <div class="fx-group-header">
          <span class="chevron">&#x25BC;</span>
          <span class="group-label">WARP</span>
          <button class="layer-fx-rand" title="Randomize"><i data-lucide="dices"></i></button>
        </div>
        <div class="fx-group-body">
          <div class="param-row" data-param="wave_amp">
            <label>Wave Amt</label>
            <input type="range" min="0" max="0.1" step="0.001" value="${layer.wave_amp}">
            <span class="value">${formatValue(layer.wave_amp, 0, 0.1, 0.001)}</span>
          </div>
          <div class="param-row" data-param="wave_freq">
            <label>Wave Freq</label>
            <input type="range" min="0" max="50" step="1" value="${layer.wave_freq}">
            <span class="value">${formatValue(layer.wave_freq, 0, 50, 1)}</span>
          </div>
          <div class="param-row" data-param="wave_speed">
            <label>Wave Speed</label>
            <input type="range" min="0" max="10" step="0.1" value="${layer.wave_speed}">
            <span class="value">${formatValue(layer.wave_speed, 0, 10, 0.1)}</span>
          </div>
          <div class="param-row select-row" data-param="wave_axis">
            <label>Wave Axis</label>
            <select>
              <option value="0" ${layer.wave_axis === 0 ? 'selected' : ''}>Horizontal</option>
              <option value="1" ${layer.wave_axis === 1 ? 'selected' : ''}>Vertical</option>
              <option value="2" ${layer.wave_axis === 2 ? 'selected' : ''}>Both</option>
            </select>
          </div>
          <div class="param-row" data-param="swirl_angle">
            <label>Swirl</label>
            <input type="range" min="-720" max="720" step="5" value="${layer.swirl_angle}">
            <span class="value">${formatValue(layer.swirl_angle, -720, 720, 5)}</span>
          </div>
          <div class="param-row" data-param="swirl_radius">
            <label>Swirl Rad</label>
            <input type="range" min="0" max="1" step="0.01" value="${layer.swirl_radius}">
            <span class="value">${formatValue(layer.swirl_radius, 0, 1, 0.01)}</span>
          </div>
          <div class="param-row" data-param="bulge_strength">
            <label>Bulge</label>
            <input type="range" min="-1" max="1" step="0.01" value="${layer.bulge_strength}">
            <span class="value">${formatValue(layer.bulge_strength, -1, 1, 0.01)}</span>
          </div>
          <div class="param-row" data-param="bulge_radius">
            <label>Bulge Rad</label>
            <input type="range" min="0.05" max="1" step="0.01" value="${layer.bulge_radius}">
            <span class="value">${formatValue(layer.bulge_radius, 0.05, 1, 0.01)}</span>
          </div>
        </div>
      </div>

      <div class="fx-group collapsed" data-layer-group="key">
        <div class="fx-group-header">
          <span class="chevron">&#x25BC;</span>
          <span class="group-label">KEY</span>
        </div>
        <div class="fx-group-body">
          <div class="param-row toggle-row" data-param="chroma_enable">
            <label>Enable</label>
            <label class="toggle"><input type="checkbox" ${layer.chroma_enable ? 'checked' : ''}><span class="toggle-slider"></span></label>
          </div>
          <div class="param-row color-row" data-param="chroma_color">
            <label>Key Color</label>
            <input type="color" value="${layer.chroma_color || '#00ff00'}">
          </div>
          <div class="param-row" data-param="chroma_threshold">
            <label>Threshold</label>
            <input type="range" min="0" max="1" step="0.01" value="${layer.chroma_threshold}">
            <span class="value">${formatValue(layer.chroma_threshold, 0, 1, 0.01)}</span>
          </div>
          <div class="param-row" data-param="chroma_smoothness">
            <label>Smoothness</label>
            <input type="range" min="0" max="1" step="0.01" value="${layer.chroma_smoothness}">
            <span class="value">${formatValue(layer.chroma_smoothness, 0, 1, 0.01)}</span>
          </div>
          <div class="param-row" data-param="chroma_spill">
            <label>Spill</label>
            <input type="range" min="0" max="1" step="0.01" value="${layer.chroma_spill}">
            <span class="value">${formatValue(layer.chroma_spill, 0, 1, 0.01)}</span>
          </div>
          <p class="layer-hint">Reveals layers below — use on upper layers.</p>
        </div>
      </div>

      <div class="fx-group collapsed" data-layer-group="shift">
        <div class="fx-group-header">
          <span class="chevron">&#x25BC;</span>
          <span class="group-label">SHIFT</span>
          <button class="layer-fx-rand" title="Randomize"><i data-lucide="dices"></i></button>
        </div>
        <div class="fx-group-body">
          <div class="param-row" data-param="slice_intensity">
            <label>Slice Amt</label>
            <input type="range" min="0" max="1" step="0.01" value="${layer.slice_intensity}">
            <span class="value">${formatValue(layer.slice_intensity, 0, 1, 0.01)}</span>
          </div>
          <div class="param-row" data-param="slice_height">
            <label>Slice H</label>
            <input type="range" min="1" max="128" step="1" value="${layer.slice_height}">
            <span class="value">${formatValue(layer.slice_height, 1, 128, 1)}</span>
          </div>
          <div class="param-row" data-param="slice_prob">
            <label>Slice Prob</label>
            <input type="range" min="0" max="1" step="0.01" value="${layer.slice_prob}">
            <span class="value">${formatValue(layer.slice_prob, 0, 1, 0.01)}</span>
          </div>
          <div class="param-row" data-param="slice_speed">
            <label>Slice Spd</label>
            <input type="range" min="0" max="30" step="1" value="${layer.slice_speed}">
            <span class="value">${formatValue(layer.slice_speed, 0, 30, 1)}</span>
          </div>
          <div class="param-row" data-param="slice_axis" title="0 = horizontal, 1 = vertical, 2 = both">
            <label>Slice Axis</label>
            <input type="range" min="0" max="2" step="1" value="${layer.slice_axis}">
            <span class="value">${formatValue(layer.slice_axis, 0, 2, 1)}</span>
          </div>
          <div class="param-row" data-param="block_size">
            <label>Block Size</label>
            <input type="range" min="4" max="128" step="1" value="${layer.block_size}">
            <span class="value">${formatValue(layer.block_size, 4, 128, 1)}</span>
          </div>
          <div class="param-row" data-param="block_intensity">
            <label>Block Amt</label>
            <input type="range" min="0" max="1" step="0.01" value="${layer.block_intensity}">
            <span class="value">${formatValue(layer.block_intensity, 0, 1, 0.01)}</span>
          </div>
          <div class="param-row" data-param="block_prob">
            <label>Block Prob</label>
            <input type="range" min="0" max="1" step="0.01" value="${layer.block_prob}">
            <span class="value">${formatValue(layer.block_prob, 0, 1, 0.01)}</span>
          </div>
          <div class="param-row" data-param="block_speed">
            <label>Block Spd</label>
            <input type="range" min="0" max="30" step="1" value="${layer.block_speed}">
            <span class="value">${formatValue(layer.block_speed, 0, 30, 1)}</span>
          </div>
          <div class="param-row" data-param="shift_chroma">
            <label>Chroma</label>
            <input type="range" min="0" max="1" step="0.01" value="${layer.shift_chroma}">
            <span class="value">${formatValue(layer.shift_chroma, 0, 1, 0.01)}</span>
          </div>
          <div class="param-row" data-param="jitter_amount">
            <label>Jitter</label>
            <input type="range" min="0" max="1" step="0.01" value="${layer.jitter_amount}">
            <span class="value">${formatValue(layer.jitter_amount, 0, 1, 0.01)}</span>
          </div>
          <div class="param-row" data-param="jitter_speed">
            <label>Jitter Spd</label>
            <input type="range" min="0" max="30" step="1" value="${layer.jitter_speed}">
            <span class="value">${formatValue(layer.jitter_speed, 0, 30, 1)}</span>
          </div>
          <div class="param-row" data-param="datamosh" title="displaced blocks bleed the previous frame (smear trails)">
            <label>Datamosh</label>
            <input type="range" min="0" max="1" step="0.01" value="${layer.datamosh}">
            <span class="value">${formatValue(layer.datamosh, 0, 1, 0.01)}</span>
          </div>
        </div>
      </div>

      <div class="fx-group collapsed" data-layer-group="transform">
        <div class="fx-group-header">
          <span class="chevron">&#x25BC;</span>
          <span class="group-label">POSITION &amp; SIZE</span>
          <button class="layer-fx-rand" title="Randomize"><i data-lucide="dices"></i></button>
        </div>
        <div class="fx-group-body">
          <div class="param-row" data-param="layer_x">
            <label>X</label>
            <input type="range" min="-1" max="1" step="0.01" value="${layer.layer_x}">
            <span class="value">${formatValue(layer.layer_x, -1, 1, 0.01)}</span>
          </div>
          <div class="param-row" data-param="layer_y">
            <label>Y</label>
            <input type="range" min="-1" max="1" step="0.01" value="${layer.layer_y}">
            <span class="value">${formatValue(layer.layer_y, -1, 1, 0.01)}</span>
          </div>
          <div class="param-row" data-param="layer_scale">
            <label>Scale</label>
            <input type="range" min="0.1" max="4" step="0.01" value="${layer.layer_scale}">
            <span class="value">${formatValue(layer.layer_scale, 0.1, 4, 0.01)}</span>
          </div>
        </div>
      </div>
    </div>
  `;

  // Playback, visibility, removal, slider input, selects, drag and group
  // collapse are all wired through delegated listeners on #layers-list, which
  // read the live card.dataset.index so they survive reordering. Here we only
  // add the two affordances delegation can't: the click-to-type value editor
  // and the ƒ modulator launcher. Both read the index live (via card.dataset
  // .index) so they keep targeting this card after it's moved.
  card.querySelectorAll('.layer-body .param-row[data-param]').forEach((row) => {
    const param = row.dataset.param;
    const slider = row.querySelector('input[type="range"]');
    const valueEl = row.querySelector('.value');
    if (!slider || !valueEl) return; // numeric rows only (skip selects/toggles/color)

    const liveIndex = () => parseInt(card.dataset.index);

    // Click the value cell to type a literal number or a fasteval formula.
    makeValueEditable(valueEl, {
      setValue: (v) => sendAction({ action: 'set_layer_param', index: liveIndex(), param, value: v }),
      setAutomation: (expr) => sendAction({ action: 'set_layer_automation', index: liveIndex(), param, expr }),
      clearAutomation: () => sendAction({ action: 'clear_layer_automation', index: liveIndex(), param }),
    });

    // …and the guided modulator builder (the ƒ launcher). `index` is a live
    // getter so the modal targets this card's current position after reorders.
    addLauncher(row, {
      scope: 'layer',
      get index() { return parseInt(card.dataset.index); },
      param,
      lo: parseFloat(slider.min), hi: parseFloat(slider.max),
      step: parseFloat(slider.step) || 0.01,
      label: (row.querySelector('label')?.textContent || param).trim(),
    });
  });

  renderIcons(card); // swap this card's <i data-lucide> placeholders for <svg>
  return card;
}

function updateLayerCard(card, layer, index) {
  if (!card) return;
  card.dataset.index = index;
  card.dataset.id = layer.id;
  const num = card.querySelector('.layer-num');
  if (num) num.textContent = index + 1;

  const title = card.querySelector('.layer-title');
  const visBtn = card.querySelector('.layer-vis-btn');
  const progressFill = card.querySelector('.layer-progress-fill');
  const thumbWrap = card.querySelector('.layer-thumb-wrap');

  if (title) title.textContent = layer.filename || 'Untitled';
  if (thumbWrap) thumbWrap.classList.toggle('paused', !!layer.paused);
  if (visBtn) {
    // Rebuild the eye <svg> only when visibility actually changes (never per-frame).
    const prev = visBtn.dataset.visible === 'true';
    if (prev !== layer.visible) {
      visBtn.dataset.visible = String(layer.visible);
      visBtn.classList.toggle('visible', layer.visible);
      visBtn.innerHTML = `<i data-lucide="${layer.visible ? 'eye' : 'eye-off'}"></i>`;
      renderIcons(visBtn);
    }
  }
  if (progressFill) {
    progressFill.style.width = `${(layer.progress * 100).toFixed(1)}%`;
  }

  // Sync every param row from the snapshot, skipping any control the user is
  // actively manipulating so we don't clobber in-progress input. Numeric rows
  // also pick up automation indicators (the ƒ launcher glow + formula tooltip).
  const autos = layer.automations;
  const errors = layer.automation_errors;
  card.querySelectorAll('.param-row[data-param]').forEach((row) => {
    const param = row.dataset.param;
    const value = layer[param];
    if (value === undefined) return;

    const slider = row.querySelector('input[type="range"]');
    const valEl = row.querySelector('.value');
    const checkbox = row.querySelector('input[type="checkbox"]');
    const select = row.querySelector('select');
    const color = row.querySelector('input[type="color"]');

    if (slider && document.activeElement !== slider) {
      slider.value = value;
      // Don't overwrite the cell while the user is typing a value/formula in it.
      if (valEl && !valEl.querySelector('input')) {
        valEl.textContent = formatValue(value, parseFloat(slider.min), parseFloat(slider.max), parseFloat(slider.step));
      }
      applyAutomationState(valEl, param, autos, errors);
    }
    if (checkbox) {
      checkbox.checked = !!value;
    }
    if (select && document.activeElement !== select) {
      select.value = (typeof value === 'string') ? value.toLowerCase() : value;
    }
    if (color && document.activeElement !== color && typeof value === 'string') {
      color.value = value;
    }
  });
}

// --- Sync library ---

// Cache for preview frame availability: filename → frame count (0 = not checked, -1 = unavailable)
const previewCache = new Map();

function syncLibrary(files) {
  if (!files) return;

  // Only rebuild if changed
  const currentCount = libraryGrid.querySelectorAll('.library-item').length;
  if (currentCount === files.length) return;

  libraryGrid.innerHTML = '';

  if (files.length === 0) {
    libraryGrid.innerHTML = '<p class="dim" style="grid-column:1/-1;text-align:center;padding:12px;">No media files</p>';
    return;
  }

  files.forEach((filename) => {
    const item = document.createElement('div');
    item.className = 'library-item';
    item.title = filename;

    // Thumbnail image from server (retries if not yet generated)
    const img = document.createElement('img');
    img.dataset.retries = '0';
    const thumbUrl = `/thumb/${encodeURIComponent(filename)}`;
    img.src = thumbUrl;
    img.onerror = () => {
      const retries = parseInt(img.dataset.retries);
      if (retries < 5) {
        img.dataset.retries = String(retries + 1);
        setTimeout(() => { img.src = `/thumb/${encodeURIComponent(filename)}?r=${retries + 1}`; }, 1500 * (retries + 1));
      } else {
        img.style.display = 'none';
        const placeholder = document.createElement('span');
        placeholder.className = 'lib-placeholder';
        placeholder.textContent = filename.replace(/\.[^.]+$/, '');
        item.appendChild(placeholder);
      }
    };
    item.appendChild(img);

    // Hover preview animation. Stills (PNG/JPG) have no /preview frames, so
    // probe frame 0 first and only cycle if it actually loads — otherwise the
    // 404s would trip the thumbnail's onerror retry chain and hide it.
    let hoverInterval = null;
    let hoverFrame = 0;
    let hovering = false;

    item.addEventListener('mouseenter', () => {
      hovering = true;
      const enc = encodeURIComponent(filename);
      const probe = new Image();
      probe.onload = () => {
        if (!hovering) return; // left before the probe resolved
        hoverFrame = 0;
        hoverInterval = setInterval(() => {
          hoverFrame = (hoverFrame + 1) % 8;
          img.src = `/preview/${enc}/${hoverFrame}`;
        }, 250);
      };
      probe.src = `/preview/${enc}/0`; // 404 for stills → never cycles, thumb stays
    });

    item.addEventListener('mouseleave', () => {
      hovering = false;
      if (hoverInterval) {
        clearInterval(hoverInterval);
        hoverInterval = null;
      }
      // Restore static thumbnail
      img.src = thumbUrl;
    });

    // Filename label on hover
    const label = document.createElement('span');
    label.className = 'lib-label';
    label.textContent = filename.replace(/\.[^.]+$/, '');
    item.appendChild(label);

    item.addEventListener('dblclick', () => {
      sendAction({ action: 'add_layer', filename });
    });

    libraryGrid.appendChild(item);
  });
}

// --- Sync patches ---

let lastPatchesKey = null;

function syncPatches(patches) {
  if (!patches) return;

  // Only rebuild when the set of names changes (avoids clobbering hover state).
  const key = patches.join('\n');
  if (key === lastPatchesKey) return;
  lastPatchesKey = key;

  patchesList.innerHTML = '';

  if (patches.length === 0) {
    patchesList.innerHTML = '<p class="dim" style="padding:6px 8px;">No saved patches</p>';
    return;
  }

  patches.forEach((name) => {
    const row = document.createElement('div');
    row.className = 'patch-row';

    const label = document.createElement('span');
    label.className = 'patch-name';
    label.textContent = name;
    label.title = `Load "${name}"`;
    label.addEventListener('click', () => {
      sendAction({ action: 'load_patch', name });
    });
    row.appendChild(label);

    const del = document.createElement('button');
    del.className = 'patch-del';
    del.textContent = '×';
    del.title = `Delete "${name}"`;
    del.addEventListener('click', (e) => {
      e.stopPropagation();
      if (confirm(`Delete patch "${name}"?`)) {
        sendAction({ action: 'delete_patch', name });
      }
    });
    row.appendChild(del);

    patchesList.appendChild(row);
  });
}

// --- Save patch (inline name field + save icon) ---

function savePatch() {
  const input = document.getElementById('patch-name');
  const name = input.value.trim();
  if (!name) return;
  sendAction({ action: 'save_patch', name });
  input.value = '';
}

document.getElementById('patch-save').addEventListener('click', savePatch);
document.getElementById('patch-name').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') {
    e.preventDefault();
    savePatch();
  }
});

// --- Sync transport ---

function syncTransport(paused) {
  const btn = document.getElementById('btn-play-all');
  btn.textContent = paused ? '\u25B6' : '\u23F8';
  btn.title = paused ? 'Play All' : 'Pause All';
}


// --- Export / Render ---

let exportActive = false;

document.getElementById('export-start').addEventListener('click', () => {
  // Export resolution follows the master output size (one source of truth).
  const duration = parseFloat(document.getElementById('export-duration').value) || 10;
  const fps = parseInt(document.getElementById('export-fps').value) || 30;
  sendAction({ action: 'start_export', width: lastOutputW, height: lastOutputH, fps, duration_secs: duration });
});

document.getElementById('export-cancel').addEventListener('click', () => {
  sendAction({ action: 'cancel_export' });
});

function syncExport(progress, error) {
  const startBtn = document.getElementById('export-start');
  const cancelBtn = document.getElementById('export-cancel');
  const progressEl = document.getElementById('export-progress');
  const fillEl = document.getElementById('export-fill');
  const textEl = document.getElementById('export-text');
  const statusEl = document.getElementById('export-status');

  if (progress > 0 && progress < 1) {
    // Rendering in progress
    exportActive = true;
    startBtn.style.display = 'none';
    cancelBtn.style.display = '';
    progressEl.style.display = '';
    fillEl.style.width = (progress * 100) + '%';
    textEl.textContent = Math.round(progress * 100) + '%';
    statusEl.textContent = '';
  } else if (progress >= 1) {
    // Done
    if (exportActive) {
      startBtn.style.display = '';
      cancelBtn.style.display = 'none';
      progressEl.style.display = 'none';
      if (error) {
        statusEl.textContent = 'Error: ' + error;
        statusEl.className = 'export-status error';
      } else {
        statusEl.textContent = 'Render complete!';
        statusEl.className = 'export-status success';
      }
      exportActive = false;
    }
  } else {
    // Idle
    if (!exportActive) {
      startBtn.style.display = '';
      cancelBtn.style.display = 'none';
      progressEl.style.display = 'none';
    }
  }
}

// --- Helpers ---

function formatValue(v, min, max, step) {
  if (step >= 1) return v.toFixed(0);
  if (max <= 1 && min >= -1) return v.toFixed(2);
  if (step >= 0.01) return v.toFixed(1);
  return v.toFixed(3);
}

// ===========================================================================
// Automation Editor (the per-param ƒ launcher)
// ===========================================================================
//
// A guided modulator builder (docs/automation-editor.md). Each automatable
// param row carries a small `ƒ` launcher; clicking it opens a modal locked to
// that scope + param, with a rack of knobs (shape / sync / rate / depth /
// center / phase-or-seed / invert) that compile live to a fasteval expression.
// Apply installs it via set_automation / set_layer_automation; Remove clears
// it; Copy puts the generated expression on the clipboard. A raw-formula
// escape hatch lets you hand-edit the expression.
//
// No backend change: the engine only runs a closed-form string, so the builder
// GENERATES a formula rather than a drawn curve. The preview is JS-side and
// mirrors the same helpers as the engine (src/automation.rs).

// --- JS mirrors of the Rust automation helpers (src/automation.rs) ---
const PI = Math.PI;
const TAU = Math.PI * 2;
const fract = (x) => x - Math.floor(x);
const tri = (t) => 1 - 4 * Math.abs(fract(t) - 0.5);          // -1..1, period 1
const saw = (t) => 2 * fract(t) - 1;                          // -1..1, period 1
const square = (t) => (fract(t) < 0.5 ? 1 : -1);              // ±1, period 1
const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);
const lerp = (a, b, t) => a + (b - a) * t;
function smoothstep(lo, hi, v) {
  if (Math.abs(hi - lo) < 1e-9) return 0;
  const x = clamp((v - lo) / (hi - lo), 0, 1);
  return x * x * (3 - 2 * x);
}
// Approximate value-noise for previews (the Rust version uses a bit-hash; this
// is close enough to convey the shape and stays deterministic).
function hash01(x) {
  const s = Math.sin(x * 127.1) * 43758.5453;
  return (s - Math.floor(s)) * 2 - 1;
}
function valueNoise(x) {
  const i = Math.floor(x);
  const f = x - i;
  const u = f * f * (3 - 2 * f);
  return lerp(hash01(i), hash01(i + 1), u);
}
const wiggle = (freq, amp, t) => valueNoise(t * freq) * amp;
const noise = (seed) => valueNoise(seed);

// The modulator model (docs/automation-editor.md §3). A small set of controls
// compiles deterministically to one fasteval expression — mental model: a synth
// LFO. Two shape families: periodic (sine/triangle/saw/square — loop exactly)
// and random (smooth/stepped — wander via the engine's deterministic value
// noise, never repeating; see §3.4).

function defaultControls() {
  return { shape: 'sine', sync: 'beat', rate: 1, depth: 1, center: 0.5, phase: 0, seed: 0, invert: false };
}

// [value, glyph, label]. The glyph is drawn on the shape button.
const SHAPES = [
  ['sine', '\u223F', 'Sine'],
  ['triangle', '\u25B3', 'Triangle'],
  ['sawup', '\u2571', 'Saw up'],
  ['sawdown', '\u2572', 'Saw down'],
  ['square', '\u2293', 'Square'],
  ['smooth', '\u2248', 'Smooth random'],
  ['stepped', '\u25A5', 'Stepped random'],
];
// Random shapes wander instead of looping: phase is meaningless, so the Phase
// control becomes Seed (picks which fixed noise sequence plays).
const RANDOM_SHAPES = new Set(['smooth', 'stepped']);

// Rate options. `value` is cycles-per-unit (per beat when synced, per second
// when free); the label names the resulting period. 1 cycle/beat == "1 beat".
const RATE_BEAT = [
  ['0.0625', '4 bars'],
  ['0.125', '2 bars'],
  ['0.25', '1 bar'],
  ['0.5', '1/2 note'],
  ['1', '1 beat'],
  ['2', '1/2 beat'],
  ['4', '1/4 beat'],
];
const RATE_FREE = [
  ['0.0625', '16 s'],
  ['0.125', '8 s'],
  ['0.25', '4 s'],
  ['0.5', '2 s'],
  ['1', '1 s'],
  ['2', '0.5 s'],
];

// Presets seed the whole knob rack from a named starting point; you then tweak.
// Each reproduces a familiar move (these mirror the old cookbook cards).
const PRESETS = [
  ['Beat pulse', { shape: 'sine', sync: 'beat', rate: 1, depth: 1, center: 0.5, phase: 0, invert: false }],
  ['Beat strobe', { shape: 'square', sync: 'beat', rate: 1, depth: 1, center: 0.5, phase: 0, invert: false }],
  ['Off-beat', { shape: 'square', sync: 'beat', rate: 0.5, depth: 1, center: 0.5, phase: 0, invert: false }],
  ['Bar sweep', { shape: 'sine', sync: 'beat', rate: 0.25, depth: 1, center: 0.5, phase: 0, invert: false }],
  ['Beat ramp', { shape: 'sawup', sync: 'beat', rate: 1, depth: 1, center: 0.5, phase: 0, invert: false }],
  ['Slow drift', { shape: 'sine', sync: 'free', rate: 0.0625, depth: 1, center: 0.5, phase: 0, invert: false }],
  ['Wander', { shape: 'smooth', sync: 'free', rate: 0.5, depth: 1, center: 0.5, seed: 0, invert: false }],
  ['Jitter', { shape: 'stepped', sync: 'beat', rate: 2, depth: 1, center: 0.5, seed: 0, invert: false }],
];

// Compact a number for a generated expression (drops float noise).
function num(n) {
  return String(parseFloat(n.toFixed(6)));
}

// Compile the control rack to a fasteval expression (docs/automation-editor.md
// §3.2). The shape S(x) is bipolar (-1..1); it's folded to 0..1 with depth +
// center, then range-scaled to [lo,hi]. Simplifications: drop +phase when 0,
// skip the clamp/center wrapper at depth 100% / center 50%, omit the range
// wrapper when the param is already 0..1. Only sine needs `x` parenthesized
// (the `*tau` multiply); tri/saw/square take raw x.
function buildExpr(c, lo, hi) {
  const base = c.sync === 'beat' ? 'beat' : 't';
  const core = c.rate === 1 ? base : `${base}*${num(c.rate)}`;
  let S;
  if (c.shape === 'smooth') {
    // Continuous value-noise → organic, non-looping drift; seed shifts which
    // stretch of noise plays (see §3.4). noise() is smooth in the engine.
    S = `noise(${c.seed ? `${core}+${num(c.seed)}` : core})`;
  } else if (c.shape === 'stepped') {
    // Held value that re-rolls on each integer step of the timebase.
    const inner = c.seed ? `floor(${core})+${num(c.seed)}` : `floor(${core})`;
    S = `noise(${inner})`;
  } else {
    const phase01 = c.phase / 360;
    const x = phase01 > 1e-6 ? `${core}+${num(phase01)}` : core;
    const compound = c.rate !== 1 || phase01 > 1e-6;
    const xw = compound ? `(${x})` : x;
    switch (c.shape) {
      case 'triangle': S = `tri(${x})`; break;
      case 'sawup': S = `saw(${x})`; break;
      case 'sawdown': S = `0-saw(${x})`; break;
      case 'square': S = `square(${x})`; break;
      default: S = `sin(${xw}*tau)`; // sine
    }
  }
  if (c.invert) S = `0-(${S})`;
  let out;
  if (Math.abs(c.depth - 1) < 1e-6 && Math.abs(c.center - 0.5) < 1e-6) {
    out = `(${S})*0.5+0.5`;
  } else {
    out = `clamp(${num(c.center)}+(${S})*${num(c.depth / 2)},0,1)`;
  }
  if (lo === 0 && hi === 1) return out;
  if (lo === 0) return `(${out})*${num(hi)}`;
  return `${num(lo)}+(${out})*${num(hi - lo)}`;
}

// JS mirror of buildExpr for the live preview — returns the normalized 0..1
// shape (drawCurve auto-scales Y, so no range-scaling needed here). Uses the
// same helpers as the engine so preview ≈ export.
function evalControls(c, tSec, beat) {
  const base = c.sync === 'beat' ? beat : tSec;
  const coreVal = base * c.rate;
  let S;
  if (c.shape === 'smooth') { S = valueNoise(coreVal + c.seed); }
  else if (c.shape === 'stepped') { S = valueNoise(Math.floor(coreVal) + c.seed); }
  else {
    const x = coreVal + c.phase / 360;
    switch (c.shape) {
      case 'triangle': S = tri(x); break;
      case 'sawup': S = saw(x); break;
      case 'sawdown': S = -saw(x); break;
      case 'square': S = square(x); break;
      default: S = Math.sin(x * TAU); // sine
    }
  }
  if (c.invert) S = -S;
  return clamp(c.center + S * (c.depth / 2), 0, 1);
}

// Live tempo, mirrored from the state snapshot (syncTempo, ~line 300). The
// preview playhead and beat-synced sampling ride these.
let liveBpm = 120;
let liveBeat = 0;

// The modal is locked to one param, set by the launcher you click:
//   aeTarget = { scope:'master'|'layer', index?, param, lo, hi, label }
let aeTarget = null;
let aeValueEl = null;                 // the row's value cell (holds dataset.expr)
let aeControls = defaultControls();
let aeRawMode = false;                // raw escape-hatch active?
let aeRaf = null;                     // preview animation-frame handle
let aePreviewCard = null;             // { canvas, ctx, fn, xUnit } for drawCurve

function isEditorOpen() {
  const m = document.getElementById('automation');
  return m && !m.hasAttribute('hidden');
}

// Add the per-param ƒ launcher to a row, between the slider and the value cell.
// `target` is the locked context the modal opens with.
function addLauncher(row, target) {
  if (!row || row.querySelector('.fx-launcher')) return;
  const valueEl = row.querySelector('.value');
  if (!valueEl) return;
  const btn = document.createElement('button');
  btn.className = 'fx-launcher';
  btn.type = 'button';
  btn.textContent = '\u0192';         // ƒ
  btn.title = 'Automate this parameter';
  btn.addEventListener('click', (e) => {
    e.stopPropagation();              // don't toggle the group / layer header
    openEditor(target, valueEl);
  });
  row.classList.add('has-fx');        // widen the grid for the launcher column
  row.insertBefore(btn, valueEl);
}

function scopeName(t) {
  return t.scope === 'master' ? 'Master' : `Layer ${t.index + 1}`;
}

// Open the modal locked to a param. If it already has a formula we can't
// reverse-parse arbitrary expressions (§5), so show it in raw mode; otherwise
// start from default controls.
function openEditor(target, valueEl) {
  aeTarget = target;
  aeValueEl = valueEl;
  aeControls = defaultControls();

  document.getElementById('ae-context').textContent =
    `Automate \u00B7 ${scopeName(target)} \u203A ${target.label}`;

  setupGhost();
  syncControlsUI();
  const existing = valueEl && valueEl.dataset.expr;
  if (existing) {
    enterRawMode(existing);           // knobs detached until a shape is picked
  } else {
    exitRawMode();
    recompute();
  }

  document.getElementById('automation').removeAttribute('hidden');
  startPreview();
}

function closeEditor() {
  const m = document.getElementById('automation');
  if (m) m.setAttribute('hidden', '');
  stopPreview();
}

// --- Control rack ↔ UI sync ---

// Reflect aeControls onto every input. Called on open, preset, shape, sync.
function syncControlsUI() {
  document.querySelectorAll('#ae-shapes .ae-shape').forEach((b) => {
    b.classList.toggle('active', b.dataset.shape === aeControls.shape);
  });
  document.querySelectorAll('#ae-sync button').forEach((b) => {
    b.classList.toggle('active', b.dataset.sync === aeControls.sync);
  });

  // Rate options follow the sync mode (cycles/beat vs cycles/sec).
  populateRate(aeControls.sync);
  const rateSel = document.getElementById('ae-rate');
  rateSel.value = String(aeControls.rate);
  if (rateSel.selectedIndex < 0) {       // rate not in the list — add it transiently
    const opt = document.createElement('option');
    opt.value = String(aeControls.rate);
    opt.textContent = String(aeControls.rate);
    rateSel.appendChild(opt);
    rateSel.value = String(aeControls.rate);
  }

  // Depth / Center are stored 0..1, shown as %.
  const depth = document.getElementById('ae-depth');
  depth.value = String(Math.round(aeControls.depth * 100));
  document.getElementById('ae-depth-val').textContent = `${Math.round(aeControls.depth * 100)}%`;
  const center = document.getElementById('ae-center');
  center.value = String(Math.round(aeControls.center * 100));
  document.getElementById('ae-center-val').textContent = `${Math.round(aeControls.center * 100)}%`;

  document.getElementById('ae-invert').checked = !!aeControls.invert;

  // Tap/tempo only matters for beat-synced formulas — grey it out in Free mode.
  const tempoRow = document.getElementById('ae-tempo-row');
  if (tempoRow) tempoRow.classList.toggle('ae-dim', aeControls.sync !== 'beat');

  morphControls();                       // Phase vs Seed, shape-dependent
}

// The bottom-left slot is Phase for periodic shapes, Seed for random ones.
function morphControls() {
  const isRandom = RANDOM_SHAPES.has(aeControls.shape);
  const lbl = document.getElementById('ae-phase-lbl');
  const slider = document.getElementById('ae-phase');
  const val = document.getElementById('ae-phase-val');
  if (isRandom) {
    lbl.textContent = 'Seed';
    slider.min = '0'; slider.max = '64'; slider.step = '1';
    slider.value = String(aeControls.seed || 0);
    val.textContent = String(aeControls.seed || 0);
  } else {
    lbl.textContent = 'Phase';
    slider.min = '0'; slider.max = '360'; slider.step = '1';
    slider.value = String(aeControls.phase || 0);
    val.textContent = `${aeControls.phase || 0}\u00B0`;
  }
}

function populateRate(sync) {
  const sel = document.getElementById('ae-rate');
  const opts = sync === 'beat' ? RATE_BEAT : RATE_FREE;
  sel.innerHTML = '';
  opts.forEach(([value, label]) => {
    const opt = document.createElement('option');
    opt.value = value;
    opt.textContent = label;
    sel.appendChild(opt);
  });
}

// --- Formula generation / raw mode ---

function currentBuiltExpr() {
  return buildExpr(aeControls, aeTarget.lo, aeTarget.hi);
}

// What Apply / Copy install: the raw text when detached, else the built one.
function currentExpr() {
  if (aeRawMode) return document.getElementById('ae-raw-input').value.trim();
  return currentBuiltExpr();
}

// Enter raw mode: show a text field prefilled with `text`. The knobs no longer
// drive the formula until a shape/preset is picked (which calls exitRawMode).
function enterRawMode(text) {
  aeRawMode = true;
  const input = document.getElementById('ae-raw-input');
  input.value = text;
  input.hidden = false;
  document.getElementById('ae-formula').hidden = true;
  document.getElementById('ae-editraw').textContent = 'Use knobs';
  document.getElementById('automation').classList.add('raw');
}

// Leave raw mode (back to knob-driven). Does NOT recompute — callers do that
// after they finish mutating aeControls, to avoid double work / ordering bugs.
function exitRawMode() {
  aeRawMode = false;
  document.getElementById('ae-raw-input').hidden = true;
  document.getElementById('ae-formula').hidden = false;
  document.getElementById('ae-editraw').textContent = 'Edit raw';
  document.getElementById('automation').classList.remove('raw');
}

// Refresh the displayed formula from the current controls (knob mode only).
function recompute() {
  if (aeRawMode) return;
  document.getElementById('ae-formula').textContent = currentBuiltExpr();
}

// --- Static UI: shape buttons + preset chips (built once on load) ---

function buildShapeButtons() {
  const box = document.getElementById('ae-shapes');
  if (!box || box.children.length) return;
  SHAPES.forEach(([value, glyph, label]) => {
    const b = document.createElement('button');
    b.className = 'ae-shape';
    b.type = 'button';
    b.dataset.shape = value;
    b.textContent = glyph;
    b.title = label;
    b.addEventListener('click', () => {
      aeControls.shape = value;
      exitRawMode();
      syncControlsUI();
      recompute();
    });
    box.appendChild(b);
  });
}

function buildPresets() {
  const box = document.getElementById('ae-presets');
  if (!box || box.children.length) return;
  PRESETS.forEach(([name, ctrl]) => {
    const b = document.createElement('button');
    b.className = 'ae-preset';
    b.type = 'button';
    b.textContent = name;
    b.addEventListener('click', () => {
      aeControls = Object.assign(defaultControls(), ctrl);
      exitRawMode();
      syncControlsUI();
      recompute();
    });
    box.appendChild(b);
  });
}

// --- Live preview (rides the tapped tempo) ---

function startPreview() {
  const canvas = document.getElementById('ae-preview');
  if (!canvas) return;
  aePreviewCard = {
    canvas,
    ctx: canvas.getContext('2d'),
    fn: (t, beat) => evalControls(aeControls, t, beat),
    xUnit: aeControls.sync === 'beat' ? 'beat' : 'sec',
  };
  if (!aeRaf) aeRaf = requestAnimationFrame(drawPreview);
}

function drawPreview() {
  if (!isEditorOpen()) { aeRaf = null; return; }
  // Keep the axis/playhead matched to the current sync mode.
  aePreviewCard.xUnit = aeControls.sync === 'beat' ? 'beat' : 'sec';
  const nowSec = performance.now() / 1000;
  drawCurve(aePreviewCard, nowSec);
  updateGhost(nowSec);
  aeRaf = requestAnimationFrame(drawPreview);
}

function stopPreview() {
  if (aeRaf) { cancelAnimationFrame(aeRaf); aeRaf = null; }
}

// Point the ghost slider at the target param's real range + step, and label
// its ends. Called when the editor opens (the range is per-param).
function setupGhost() {
  const g = document.getElementById('ae-ghost');
  if (!g || !aeTarget) return;
  g.min = aeTarget.lo;
  g.max = aeTarget.hi;
  g.step = aeTarget.step || 0.01;
  document.getElementById('ae-ghost-lo').textContent = num(aeTarget.lo);
  document.getElementById('ae-ghost-hi').textContent = num(aeTarget.hi);
}

// Drive the ghost slider with the *actual* value the formula outputs right now
// (true live time/beat — not the windowed playhead), mapped into the param's
// real range. This is exactly what will be sent to the param, so it previews
// "what the slider will do" once applied.
function updateGhost(nowSec) {
  const g = document.getElementById('ae-ghost');
  if (!g || !aeTarget) return;
  const norm = clamp(evalControls(aeControls, nowSec, liveBeat), 0, 1);
  const real = aeTarget.lo + norm * (aeTarget.hi - aeTarget.lo);
  g.value = real;
  document.getElementById('ae-ghost-val').textContent =
    formatValue(real, aeTarget.lo, aeTarget.hi, parseFloat(g.step) || 0.01);
}

// --- Install / remove ---

function applyEditor() {
  if (!aeTarget) return;
  const expr = currentExpr();
  if (!expr) return;
  if (aeTarget.scope === 'master') {
    sendAction({ action: 'set_automation', param: aeTarget.param, expr });
  } else {
    sendAction({ action: 'set_layer_automation', index: aeTarget.index, param: aeTarget.param, expr });
  }
  closeEditor();
}

function removeEditor() {
  if (!aeTarget) return;
  if (aeTarget.scope === 'master') {
    sendAction({ action: 'clear_automation', param: aeTarget.param });
  } else {
    sendAction({ action: 'clear_layer_automation', index: aeTarget.index, param: aeTarget.param });
  }
  closeEditor();
}

function copyText(text, btn) {
  navigator.clipboard.writeText(text).then(() => {
    const old = btn.textContent;
    btn.textContent = 'Copied';
    setTimeout(() => { btn.textContent = old; }, 900);
  }).catch(() => {});
}

const accent = getComputedStyle(document.documentElement).getPropertyValue('--accent').trim() || '#5a8fe6';
const borderCol = getComputedStyle(document.documentElement).getPropertyValue('--border').trim() || '#2a2a35';

// Plot one formula over a 4-unit window, auto-scaling Y, with a moving playhead.
function drawCurve(card, nowSec) {
  const { canvas, ctx, fn, xUnit } = card;
  const W = canvas.width, H = canvas.height;
  const pad = 4;
  const span = 4;             // 4 beats or 4 seconds across the canvas
  const samples = 120;

  // Sample the curve across the window. Beat cards map x→beats, time cards x→sec.
  const ys = new Array(samples);
  for (let i = 0; i < samples; i++) {
    const u = (i / (samples - 1)) * span;   // 0..span
    let v;
    if (xUnit === 'beat') {
      // At the live tempo, beats convert to seconds for the `t` arg.
      const tSec = (u / liveBpm) * 60;
      v = fn(tSec, u, liveBpm);
    } else {
      v = fn(u, u * liveBpm / 60, liveBpm);
    }
    ys[i] = Number.isFinite(v) ? v : 0;
  }

  // Auto-scale Y to the sampled range with 10% padding.
  let lo = Math.min(...ys), hi = Math.max(...ys);
  if (hi - lo < 1e-6) { lo -= 0.5; hi += 0.5; }
  const range = hi - lo;
  lo -= range * 0.1; hi += range * 0.1;
  const yToPx = (v) => H - pad - ((v - lo) / (hi - lo)) * (H - 2 * pad);
  const xToPx = (i) => pad + (i / (samples - 1)) * (W - 2 * pad);

  ctx.clearRect(0, 0, W, H);

  // Baseline at y=0 if it falls within range.
  ctx.strokeStyle = borderCol;
  ctx.lineWidth = 1;
  if (lo < 0 && hi > 0) {
    const zy = yToPx(0);
    ctx.beginPath();
    ctx.moveTo(pad, zy);
    ctx.lineTo(W - pad, zy);
    ctx.stroke();
  } else {
    ctx.strokeRect(0.5, 0.5, W - 1, H - 1);
  }

  // The curve.
  ctx.strokeStyle = accent;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (let i = 0; i < samples; i++) {
    const x = xToPx(i), y = yToPx(ys[i]);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Moving playhead. Beat cards align to the live beat phase; time cards use
  // wall-clock seconds. Both wrap within the 4-unit window.
  let phase;
  if (xUnit === 'beat') {
    phase = (liveBeat % span + span) % span;
  } else {
    phase = nowSec % span;
  }
  const px = pad + (phase / span) * (W - 2 * pad);
  ctx.strokeStyle = accent;
  ctx.globalAlpha = 0.5;
  ctx.beginPath();
  ctx.moveTo(px, 0);
  ctx.lineTo(px, H);
  ctx.stroke();
  ctx.globalAlpha = 1;

  // Dot riding the curve at the playhead.
  const fi = (phase / span) * (samples - 1);
  const i0 = Math.floor(fi), i1 = Math.min(i0 + 1, samples - 1);
  const yv = lerp(ys[i0], ys[i1], fi - i0);
  ctx.fillStyle = accent;
  ctx.beginPath();
  ctx.arc(px, yToPx(yv), 2.5, 0, TAU);
  ctx.fill();
}

// --- Wire the editor (built once; launchers are added per-row at init) ---

buildShapeButtons();
buildPresets();

// Close affordances: ✕ button, backdrop click, Escape.
document.getElementById('ae-close').addEventListener('click', closeEditor);
document.getElementById('automation').addEventListener('click', (e) => {
  if (e.target.id === 'automation') closeEditor();   // backdrop, not the panel
});
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && isEditorOpen()) closeEditor();
});

// Sync segment: switching time base resets rate to a sensible default for that
// base (its options differ — cycles/beat vs cycles/sec).
document.querySelectorAll('#ae-sync button').forEach((b) => {
  b.addEventListener('click', () => {
    aeControls.sync = b.dataset.sync;
    aeControls.rate = b.dataset.sync === 'beat' ? 1 : 0.25;
    exitRawMode();
    syncControlsUI();
    recompute();
  });
});

document.getElementById('ae-rate').addEventListener('change', (e) => {
  aeControls.rate = parseFloat(e.target.value);
  exitRawMode();
  recompute();
});

document.getElementById('ae-depth').addEventListener('input', (e) => {
  aeControls.depth = parseInt(e.target.value) / 100;
  document.getElementById('ae-depth-val').textContent = `${e.target.value}%`;
  exitRawMode();
  recompute();
});

document.getElementById('ae-center').addEventListener('input', (e) => {
  aeControls.center = parseInt(e.target.value) / 100;
  document.getElementById('ae-center-val').textContent = `${e.target.value}%`;
  exitRawMode();
  recompute();
});

// The shape-dependent slider: Seed for random shapes, Phase otherwise.
document.getElementById('ae-phase').addEventListener('input', (e) => {
  const v = parseInt(e.target.value);
  if (RANDOM_SHAPES.has(aeControls.shape)) {
    aeControls.seed = v;
    document.getElementById('ae-phase-val').textContent = String(v);
  } else {
    aeControls.phase = v;
    document.getElementById('ae-phase-val').textContent = `${v}\u00B0`;
  }
  exitRawMode();
  recompute();
});

document.getElementById('ae-invert').addEventListener('change', (e) => {
  aeControls.invert = e.target.checked;
  exitRawMode();
  recompute();
});

// Edit raw ↔ knobs toggle.
document.getElementById('ae-editraw').addEventListener('click', () => {
  if (aeRawMode) { exitRawMode(); recompute(); }
  else { enterRawMode(currentBuiltExpr()); }
});

document.getElementById('ae-apply').addEventListener('click', applyEditor);
document.getElementById('ae-remove').addEventListener('click', removeEditor);
document.getElementById('ae-copy').addEventListener('click', (e) =>
  copyText(currentExpr(), e.currentTarget));

// Render Lucide icons within `el` only. Guarded so a blocked CDN degrades to
// empty (decorative) buttons instead of throwing — titles still convey function.
function renderIcons(el) {
  if (window.lucide && el) lucide.createIcons({ root: el });
}

// Pick a random value in [min, max] snapped to `step`, without float drift.
function randInRange(min, max, step) {
  const s = step > 0 ? step : 0.01;
  const steps = Math.max(1, Math.round((max - min) / s));
  const v = min + Math.round(Math.random() * steps) * s;
  const decimals = (String(s).split('.')[1] || '').length;
  return parseFloat(v.toFixed(decimals));
}

// Randomize the range sliders of one per-layer FX group (COLOR/DIGITAL).
// Sliders only — the invert toggle and any selects are left untouched.
function randomizeLayerGroup(fxGroup, index) {
  if (!fxGroup) return;
  fxGroup.querySelectorAll('.param-row[data-param]').forEach((row) => {
    const slider = row.querySelector('input[type="range"]');
    if (!slider) return;
    const min = parseFloat(slider.min);
    const max = parseFloat(slider.max);
    const step = parseFloat(slider.step) || 0.01;
    const v = randInRange(min, max, step);
    slider.value = v;
    const valEl = row.querySelector('.value');
    if (valEl) valEl.textContent = formatValue(v, min, max, step);
    sendAction({ action: 'set_layer_param', index, param: row.dataset.param, value: v });
  });
}

// Render the static (master-panel) Lucide placeholders once on load.
if (window.lucide) lucide.createIcons();

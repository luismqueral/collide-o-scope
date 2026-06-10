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
        syncEffects(msg.effects);
        syncNtsc(msg.ntsc);
        syncFramerate(msg.framerate);
        syncLayers(msg.layers);
        syncLibrary(msg.library);
        syncPatches(msg.patches || []);
        syncTransport(msg.paused);
        syncExport(msg.export_progress, msg.export_error);
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

document.querySelectorAll('.group-reset').forEach((btn) => {
  btn.addEventListener('click', (e) => {
    e.stopPropagation();
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

document.getElementById('btn-stop').addEventListener('click', () => {
  sendAction({ action: 'reset_fx' });
});

// --- Sync effects UI from server ---

function syncEffects(effects) {
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
      const min = parseFloat(row.dataset.min);
      const max = parseFloat(row.dataset.max);
      const step = parseFloat(row.dataset.step);
      valueEl.textContent = formatValue(value, min, max, step);
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
        </div>
      </div>
    </div>
  `;

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
  // actively manipulating so we don't clobber in-progress input.
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
      if (valEl) valEl.textContent = formatValue(value, parseFloat(slider.min), parseFloat(slider.max), parseFloat(slider.step));
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
    libraryGrid.innerHTML = '<p class="dim" style="grid-column:1/-1;text-align:center;padding:12px;">No video files</p>';
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

    // Hover preview animation
    let hoverInterval = null;
    let hoverFrame = 0;

    item.addEventListener('mouseenter', () => {
      const enc = encodeURIComponent(filename);
      // Start cycling through preview frames
      hoverFrame = 0;
      hoverInterval = setInterval(() => {
        hoverFrame = (hoverFrame + 1) % 8;
        img.src = `/preview/${enc}/${hoverFrame}`;
      }, 250);
    });

    item.addEventListener('mouseleave', () => {
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
  const [w, h] = document.getElementById('export-resolution').value.split('x').map(Number);
  const duration = parseFloat(document.getElementById('export-duration').value) || 10;
  const fps = parseInt(document.getElementById('export-fps').value) || 30;
  sendAction({ action: 'start_export', width: w, height: h, fps, duration_secs: duration });
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

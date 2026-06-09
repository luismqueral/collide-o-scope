// collide-o-scope — web control panel

const statusEl = document.getElementById('ws-status');
const layersList = document.getElementById('layers-list');
const layersEmpty = document.getElementById('layers-empty');
const libraryGrid = document.getElementById('library-grid');

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
        syncLayers(msg.layers);
        syncLibrary(msg.library);
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
    });
  }

  if (checkbox) {
    checkbox.addEventListener('change', () => {
      sendAction({ action: 'set_ntsc_param', param, value: checkbox.checked });
    });
  }

  if (select) {
    select.addEventListener('change', () => {
      sendAction({ action: 'set_ntsc_param', param, value: parseInt(select.value) });
    });
  }
});

// --- Collapsible FX groups ---

document.querySelectorAll('.fx-group-header').forEach((header) => {
  header.addEventListener('click', (e) => {
    if (e.target.classList.contains('group-reset')) return;
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

  if (e.target.closest('.layer-play-btn')) {
    e.stopPropagation();
    sendAction({ action: 'toggle_layer_pause', index });
  } else if (e.target.closest('.layer-vis-btn')) {
    e.stopPropagation();
    sendAction({ action: 'toggle_visibility', index });
  } else if (e.target.closest('.layer-remove-btn')) {
    e.stopPropagation();
    sendAction({ action: 'remove_layer', index });
  } else if (e.target.closest('.layer-grip')) {
    // grip is for dragging only — don't toggle expand
  } else if (e.target.closest('.layer-header')) {
    card.classList.toggle('expanded');
  }
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
  }
});

// --- Layer drag-and-drop reorder ---
// Only the `.layer-grip` handle is draggable so sliders/buttons keep working.
// Rust is authoritative for order: we send move_layer and let the next state
// push re-render. `dragging` blocks syncLayers from churning mid-drag.

let dragSrcIndex = null;
let dragging = false;

layersList.addEventListener('dragstart', (e) => {
  const grip = e.target.closest('.layer-grip');
  if (!grip) { e.preventDefault(); return; } // block default img drag etc.
  const card = grip.closest('.layer-card');
  dragSrcIndex = parseInt(card.dataset.index);
  dragging = true;
  card.classList.add('dragging');
  e.dataTransfer.effectAllowed = 'move';
  e.dataTransfer.setData('text/plain', String(dragSrcIndex)); // Firefox needs data
});

function clearDropMarkers() {
  layersList.querySelectorAll('.drop-before, .drop-after').forEach((c) => {
    c.classList.remove('drop-before', 'drop-after');
  });
}

layersList.addEventListener('dragover', (e) => {
  if (!dragging) return;
  e.preventDefault();
  e.dataTransfer.dropEffect = 'move';
  const card = e.target.closest('.layer-card');
  clearDropMarkers();
  if (!card) return;
  const rect = card.getBoundingClientRect();
  const before = (e.clientY - rect.top) < rect.height / 2;
  card.classList.add(before ? 'drop-before' : 'drop-after');
});

layersList.addEventListener('drop', (e) => {
  if (!dragging) return;
  e.preventDefault();
  const card = e.target.closest('.layer-card');
  clearDropMarkers();
  if (card) {
    const targetIndex = parseInt(card.dataset.index);
    const rect = card.getBoundingClientRect();
    const before = (e.clientY - rect.top) < rect.height / 2;
    const insertion = before ? targetIndex : targetIndex + 1;
    // Adjust for the gap left by removing the source first.
    let to = dragSrcIndex < insertion ? insertion - 1 : insertion;
    to = Math.max(0, Math.min(to, layersList.children.length - 1));
    if (to !== dragSrcIndex) {
      sendAction({ action: 'move_layer', from: dragSrcIndex, to });
    }
  }
});

layersList.addEventListener('dragend', () => {
  layersList.querySelectorAll('.dragging').forEach((c) => c.classList.remove('dragging'));
  clearDropMarkers();
  dragging = false;
  dragSrcIndex = null;
});

// --- Sync layers ---

function syncLayers(layers) {
  if (!layers) return;
  if (dragging) return; // don't reorder DOM mid-drag
  layersEmpty.style.display = layers.length === 0 ? 'block' : 'none';

  // Rebuild if count changed
  if (layersList.children.length !== layers.length) {
    layersList.innerHTML = '';
    layers.forEach((layer, i) => {
      layersList.appendChild(createLayerCard(layer, i));
    });
  } else {
    layers.forEach((layer, i) => {
      updateLayerCard(layersList.children[i], layer, i);
    });
  }
}

function createLayerCard(layer, index) {
  const card = document.createElement('div');
  card.className = 'layer-card expanded';
  card.dataset.index = index;

  card.innerHTML = `
    <div class="layer-header">
      <span class="layer-grip" draggable="true" title="Drag to reorder">\u2261</span>
      <img class="layer-thumb" src="/thumb/${encodeURIComponent(layer.filename)}" alt="">
      <span class="layer-num">${index + 1}</span>
      <button class="layer-play-btn" title="Play/Pause">${layer.paused ? '\u25B6' : '\u25A0'}</button>
      <span class="layer-title">${layer.filename || 'Untitled'}</span>
      <button class="layer-vis-btn ${layer.visible ? 'visible' : ''}" title="Visibility">${layer.visible ? '\u25C9' : '\u25CB'}</button>
      <button class="layer-remove-btn" title="Remove">\u00D7</button>
    </div>
    <div class="layer-progress"><div class="layer-progress-fill" style="width:${(layer.progress * 100).toFixed(1)}%"></div></div>
    <div class="layer-body">
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
      <div class="param-row select-row" data-param="blend_mode">
        <label>Blend</label>
        <select>
          <option value="normal" ${layer.blend_mode === 'normal' ? 'selected' : ''}>Normal</option>
          <option value="screen" ${layer.blend_mode === 'screen' ? 'selected' : ''}>Screen</option>
          <option value="multiply" ${layer.blend_mode === 'multiply' ? 'selected' : ''}>Multiply</option>
          <option value="difference" ${layer.blend_mode === 'difference' ? 'selected' : ''}>Difference</option>
        </select>
      </div>

      <div class="layer-fx-group">
        <div class="layer-fx-label">COLOR</div>
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

      <div class="layer-fx-group">
        <div class="layer-fx-label">DIGITAL</div>
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
  `;

  return card;
}

function updateLayerCard(card, layer, index) {
  if (!card) return;
  card.dataset.index = index;
  const num = card.querySelector('.layer-num');
  if (num) num.textContent = index + 1;

  const playBtn = card.querySelector('.layer-play-btn');
  const title = card.querySelector('.layer-title');
  const visBtn = card.querySelector('.layer-vis-btn');
  const progressFill = card.querySelector('.layer-progress-fill');

  if (playBtn) playBtn.textContent = layer.paused ? '\u25B6' : '\u25A0';
  if (title) title.textContent = layer.filename || 'Untitled';
  if (visBtn) {
    visBtn.textContent = layer.visible ? '\u25C9' : '\u25CB';
    visBtn.className = `layer-vis-btn ${layer.visible ? 'visible' : ''}`;
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

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
        syncLayers(msg.layers);
        syncLibrary(msg.library);
        syncTransport(msg.paused);
      }
    } catch (err) {
      console.warn('[ws] parse error:', err);
    }
  };
}
connect();

function sendAction(action) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(action));
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

// --- Sync layers ---

function syncLayers(layers) {
  if (!layers) return;
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
      <span class="layer-num">${index + 1}</span>
      <button class="layer-play-btn" title="Play/Pause">${layer.paused ? '\u25B6' : '\u25A0'}</button>
      <span class="layer-title">${layer.filename || 'Untitled'}</span>
      <button class="layer-vis-btn ${layer.visible ? 'visible' : ''}" title="Visibility">${layer.visible ? '\u25C9' : '\u25CB'}</button>
      <button class="layer-remove-btn" title="Remove">\u00D7</button>
    </div>
    <div class="layer-progress"><div class="layer-progress-fill" style="width:${(layer.progress * 100).toFixed(1)}%"></div></div>
    <div class="layer-body">
      <div class="param-row" data-layer="${index}" data-param="opacity">
        <label>Opacity</label>
        <input type="range" min="0" max="1" step="0.01" value="${layer.opacity}">
        <span class="value">${layer.opacity.toFixed(2)}</span>
      </div>
      <div class="param-row" data-layer="${index}" data-param="speed">
        <label>Speed</label>
        <input type="range" min="0.25" max="4" step="0.25" value="${layer.speed}">
        <span class="value">${layer.speed.toFixed(2)}</span>
      </div>
      <div class="param-row select-row" data-layer="${index}" data-param="blend_mode">
        <label>Blend</label>
        <select>
          <option value="normal" ${layer.blend_mode === 'normal' ? 'selected' : ''}>Normal</option>
          <option value="screen" ${layer.blend_mode === 'screen' ? 'selected' : ''}>Screen</option>
          <option value="multiply" ${layer.blend_mode === 'multiply' ? 'selected' : ''}>Multiply</option>
          <option value="difference" ${layer.blend_mode === 'difference' ? 'selected' : ''}>Difference</option>
        </select>
      </div>
    </div>
  `;

  // Toggle expand
  card.querySelector('.layer-header').addEventListener('click', (e) => {
    if (e.target.tagName === 'BUTTON') return;
    card.classList.toggle('expanded');
  });

  // Play/pause
  card.querySelector('.layer-play-btn').addEventListener('click', () => {
    sendAction({ action: 'toggle_layer_pause', index });
  });

  // Visibility
  card.querySelector('.layer-vis-btn').addEventListener('click', () => {
    sendAction({ action: 'toggle_visibility', index });
  });

  // Remove
  card.querySelector('.layer-remove-btn').addEventListener('click', () => {
    sendAction({ action: 'remove_layer', index });
  });

  // Layer param sliders
  card.querySelectorAll('.layer-body .param-row[data-param]').forEach((row) => {
    const param = row.dataset.param;
    const slider = row.querySelector('input[type="range"]');
    const valueEl = row.querySelector('.value');
    const select = row.querySelector('select');

    if (slider) {
      slider.addEventListener('input', () => {
        const v = parseFloat(slider.value);
        if (valueEl) valueEl.textContent = v.toFixed(2);
        sendAction({ action: 'set_layer_param', index, param, value: v });
      });
    }

    if (select) {
      select.addEventListener('change', () => {
        sendAction({ action: 'set_layer_param', index, param, value: select.value });
      });
    }
  });

  return card;
}

function updateLayerCard(card, layer, index) {
  if (!card) return;
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

  // Sync layer param sliders (skip if user is actively dragging)
  const opacityRow = card.querySelector('.param-row[data-param="opacity"]');
  if (opacityRow) {
    const slider = opacityRow.querySelector('input[type="range"]');
    const valEl = opacityRow.querySelector('.value');
    if (slider && document.activeElement !== slider) {
      slider.value = layer.opacity;
      if (valEl) valEl.textContent = layer.opacity.toFixed(2);
    }
  }

  const speedRow = card.querySelector('.param-row[data-param="speed"]');
  if (speedRow) {
    const slider = speedRow.querySelector('input[type="range"]');
    const valEl = speedRow.querySelector('.value');
    if (slider && document.activeElement !== slider) {
      slider.value = layer.speed;
      if (valEl) valEl.textContent = layer.speed.toFixed(2);
    }
  }

  const blendRow = card.querySelector('.param-row[data-param="blend_mode"]');
  if (blendRow) {
    const select = blendRow.querySelector('select');
    if (select && document.activeElement !== select) {
      select.value = layer.blend_mode;
    }
  }
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


// --- Helpers ---

function formatValue(v, min, max, step) {
  if (step >= 1) return v.toFixed(0);
  if (max <= 1 && min >= -1) return v.toFixed(2);
  if (step >= 0.01) return v.toFixed(1);
  return v.toFixed(3);
}

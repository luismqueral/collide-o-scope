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
        syncEffects(msg.effects, msg.automations, msg.automation_errors);
        syncNtsc(msg.ntsc);
        syncLayers(msg.layers);
        syncLibrary(msg.library);
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
      scope: 'master', param, lo: min, hi: max,
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

document.getElementById('btn-tap').addEventListener('click', tapTempo);

// `T` taps tempo too — but ignore it while typing in a field.
document.addEventListener('keydown', (e) => {
  if (e.key !== 't' && e.key !== 'T') return;
  const tag = (e.target.tagName || '').toLowerCase();
  if (tag === 'input' || tag === 'textarea' || tag === 'select') return;
  tapTempo();
});

// Click the BPM readout to type an exact tempo. Reuses the value-cell editor;
// BPM can't be automated, so the automation hooks are no-ops.
makeValueEditable(document.getElementById('bpm-readout'), {
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
  const readout = document.getElementById('bpm-readout');
  // Don't clobber the field while the user is typing a tempo.
  if (readout && !readout.querySelector('input')) {
    readout.textContent = String(Math.round(bpm));
  }
  const dot = document.getElementById('beat-dot');
  if (dot && typeof beat === 'number') {
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
      <img class="layer-thumb" src="/thumb/${encodeURIComponent(layer.filename)}" alt="">
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
  card.querySelector('.layer-play-btn').addEventListener('click', (e) => {
    e.stopPropagation();
    console.log('[layer] play/pause clicked, index:', index);
    sendAction({ action: 'toggle_layer_pause', index });
  });

  // Visibility
  card.querySelector('.layer-vis-btn').addEventListener('click', (e) => {
    e.stopPropagation();
    console.log('[layer] visibility clicked, index:', index);
    sendAction({ action: 'toggle_visibility', index });
  });

  // Remove
  card.querySelector('.layer-remove-btn').addEventListener('click', (e) => {
    e.stopPropagation();
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

      // Numeric layer params can be automated by clicking the value.
      makeValueEditable(valueEl, {
        setValue: (v) => sendAction({ action: 'set_layer_param', index, param, value: v }),
        setAutomation: (expr) => sendAction({ action: 'set_layer_automation', index, param, expr }),
        clearAutomation: () => sendAction({ action: 'clear_layer_automation', index, param }),
      });

      // …and via the guided modulator builder (the ƒ launcher).
      addLauncher(row, {
        scope: 'layer', index, param,
        lo: parseFloat(slider.min), hi: parseFloat(slider.max),
        label: (row.querySelector('label')?.textContent || param).trim(),
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

  const autos = layer.automations;
  const errors = layer.automation_errors;

  // Sync layer param sliders (skip if user is actively dragging)
  const opacityRow = card.querySelector('.param-row[data-param="opacity"]');
  if (opacityRow) {
    const slider = opacityRow.querySelector('input[type="range"]');
    const valEl = opacityRow.querySelector('.value');
    if (slider && document.activeElement !== slider) {
      slider.value = layer.opacity;
      if (valEl && !valEl.querySelector('input')) valEl.textContent = layer.opacity.toFixed(2);
      applyAutomationState(valEl, 'opacity', autos, errors);
    }
  }

  const speedRow = card.querySelector('.param-row[data-param="speed"]');
  if (speedRow) {
    const slider = speedRow.querySelector('input[type="range"]');
    const valEl = speedRow.querySelector('.value');
    if (slider && document.activeElement !== slider) {
      slider.value = layer.speed;
      if (valEl && !valEl.querySelector('input')) valEl.textContent = layer.speed.toFixed(2);
      applyAutomationState(valEl, 'speed', autos, errors);
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
  drawCurve(aePreviewCard, performance.now() / 1000);
  aeRaf = requestAnimationFrame(drawPreview);
}

function stopPreview() {
  if (aeRaf) { cancelAnimationFrame(aeRaf); aeRaf = null; }
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

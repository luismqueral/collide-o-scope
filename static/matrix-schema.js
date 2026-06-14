// matrix-schema.js — single source of truth for the transposed parameter matrix.
//
// Rows = parameters (grouped). Columns = channels (Master FX, VHS/NTSC, layers).
// This file only describes structure; matrix.js does all the DOM/behavior.
//
// ParamDef fields:
//   key         snapshot/action key (must match EffectsSnapshot / NtscSnapshot / LayerSnapshot)
//   label       short row label
//   ptype       'float' | 'bipolar' | 'enum' | 'bool' | 'color' | 'clip'
//   min,max     numeric range (floats/bipolar/enum-as-range)
//   step        editing step
//   def         default/rest value (drives the dim-vs-changed styling)
//   options     enum only: [{value, label}] — value is Number (int enum) or String (blend_mode)
//   automatable whether the ƒ automation modal applies (NTSC has NO automation → false)
//   channels    'master' | 'ntsc' | 'layer' | 'both'  (which column kinds the row applies to)

const MATRIX_GROUPS = [
  {
    name: 'LAYER',
    params: [
      { key: 'clip', label: 'clip', ptype: 'clip', automatable: false, channels: 'layer' },
      { key: 'opacity', label: 'opacity', ptype: 'float', min: 0, max: 1, step: 0.01, def: 1, automatable: true, channels: 'layer' },
      { key: 'speed', label: 'speed', ptype: 'float', min: 0.25, max: 4, step: 0.25, def: 1, automatable: true, channels: 'layer' },
      { key: 'fps', label: 'fps', ptype: 'float', min: 1, max: 30, step: 1, def: 30, automatable: true, channels: 'layer' },
      { key: 'blend_mode', label: 'blend', ptype: 'enum', def: 'normal', automatable: false, channels: 'layer',
        options: [ { value: 'normal', label: 'normal' }, { value: 'screen', label: 'screen' }, { value: 'multiply', label: 'multiply' }, { value: 'difference', label: 'difference' } ] },
      { key: 'visible', label: 'visible', ptype: 'bool', def: true, automatable: false, channels: 'layer' },
      { key: 'paused', label: 'paused', ptype: 'bool', def: false, automatable: false, channels: 'layer' },
    ],
  },
  {
    name: 'COLOR',
    params: [
      { key: 'hue_shift', label: 'hue', ptype: 'bipolar', min: -180, max: 180, step: 1, def: 0, automatable: true, channels: 'both' },
      { key: 'saturation', label: 'sat', ptype: 'bipolar', min: -1, max: 1, step: 0.01, def: 0, automatable: true, channels: 'both' },
      { key: 'brightness', label: 'bright', ptype: 'bipolar', min: -1, max: 1, step: 0.01, def: 0, automatable: true, channels: 'both' },
      { key: 'contrast', label: 'contrast', ptype: 'bipolar', min: -1, max: 1, step: 0.01, def: 0, automatable: true, channels: 'both' },
    ],
  },
  {
    name: 'DIGITAL',
    params: [
      { key: 'pixelate', label: 'pixelate', ptype: 'float', min: 1, max: 32, step: 1, def: 1, automatable: true, channels: 'both' },
      { key: 'rgb_split', label: 'rgb split', ptype: 'float', min: 0, max: 30, step: 0.5, def: 0, automatable: true, channels: 'both' },
      { key: 'posterize', label: 'posterize', ptype: 'float', min: 0, max: 16, step: 1, def: 0, automatable: true, channels: 'both' },
      { key: 'invert', label: 'invert', ptype: 'bool', def: false, automatable: false, channels: 'both' },
    ],
  },
  {
    name: 'ANALOG',
    params: [
      { key: 'grain_intensity', label: 'grain', ptype: 'float', min: 0, max: 0.3, step: 0.005, def: 0, automatable: true, channels: 'master' },
      { key: 'grain_size', label: 'grain sz', ptype: 'float', min: 1, max: 4, step: 0.25, def: 1, automatable: true, channels: 'master' },
      { key: 'grain_algo', label: 'grain algo', ptype: 'enum', def: 0, automatable: false, channels: 'master',
        options: [ { value: 0, label: 'Gaussian' }, { value: 1, label: 'Perlin' }, { value: 2, label: 'S&P' }, { value: 3, label: 'Blue' } ] },
      { key: 'color_grain', label: 'color grain', ptype: 'bool', def: false, automatable: false, channels: 'master' },
      { key: 'vignette', label: 'vignette', ptype: 'float', min: 0, max: 1.5, step: 0.01, def: 0, automatable: true, channels: 'master' },
      { key: 'color_drift', label: 'color drift', ptype: 'float', min: 0, max: 0.02, step: 0.001, def: 0, automatable: true, channels: 'master' },
    ],
  },
  {
    name: 'MOTION',
    params: [
      { key: 'breathe_scale', label: 'breathe sc', ptype: 'float', min: 0, max: 0.05, step: 0.001, def: 0, automatable: true, channels: 'master' },
      { key: 'breathe_rotation', label: 'breathe rot', ptype: 'float', min: 0, max: 2, step: 0.05, def: 0, automatable: true, channels: 'master' },
      { key: 'breathe_position', label: 'breathe pos', ptype: 'float', min: 0, max: 0.02, step: 0.001, def: 0, automatable: true, channels: 'master' },
    ],
  },
  {
    name: 'WARP',
    params: [
      { key: 'wave_amp', label: 'wave amp', ptype: 'float', min: 0, max: 0.1, step: 0.001, def: 0, automatable: true, channels: 'layer' },
      { key: 'wave_freq', label: 'wave freq', ptype: 'float', min: 0, max: 50, step: 1, def: 0, automatable: true, channels: 'layer' },
      { key: 'wave_speed', label: 'wave spd', ptype: 'float', min: 0, max: 10, step: 0.1, def: 0, automatable: true, channels: 'layer' },
      { key: 'wave_axis', label: 'wave axis', ptype: 'enum', def: 0, automatable: false, channels: 'layer',
        options: [ { value: 0, label: 'H' }, { value: 1, label: 'V' }, { value: 2, label: 'Both' } ] },
      { key: 'swirl_angle', label: 'swirl ang', ptype: 'bipolar', min: -720, max: 720, step: 5, def: 0, automatable: true, channels: 'layer' },
      { key: 'swirl_radius', label: 'swirl rad', ptype: 'float', min: 0, max: 1, step: 0.01, def: 0, automatable: true, channels: 'layer' },
      { key: 'bulge_strength', label: 'bulge str', ptype: 'bipolar', min: -1, max: 1, step: 0.01, def: 0, automatable: true, channels: 'layer' },
      { key: 'bulge_radius', label: 'bulge rad', ptype: 'float', min: 0.05, max: 1, step: 0.01, def: 0.05, automatable: true, channels: 'layer' },
    ],
  },
  {
    name: 'KEY',
    params: [
      { key: 'chroma_enable', label: 'key on', ptype: 'bool', def: false, automatable: false, channels: 'layer' },
      { key: 'chroma_color', label: 'key color', ptype: 'color', def: '#00ff00', automatable: false, channels: 'layer' },
      { key: 'chroma_threshold', label: 'threshold', ptype: 'float', min: 0, max: 1, step: 0.01, def: 0.4, automatable: true, channels: 'layer' },
      { key: 'chroma_smoothness', label: 'smoothness', ptype: 'float', min: 0, max: 1, step: 0.01, def: 0.1, automatable: true, channels: 'layer' },
      { key: 'chroma_spill', label: 'spill', ptype: 'float', min: 0, max: 1, step: 0.01, def: 0, automatable: true, channels: 'layer' },
      { key: 'chroma_bg_enable', label: 'bg on', ptype: 'bool', def: false, automatable: false, channels: 'layer' },
      { key: 'chroma_bg_color', label: 'bg color', ptype: 'color', def: '#000000', automatable: false, channels: 'layer' },
    ],
  },
  {
    name: 'SHIFT',
    params: [
      { key: 'slice_intensity', label: 'slice int', ptype: 'float', min: 0, max: 1, step: 0.01, def: 0, automatable: true, channels: 'layer' },
      { key: 'slice_height', label: 'slice ht', ptype: 'float', min: 1, max: 128, step: 1, def: 1, automatable: true, channels: 'layer' },
      { key: 'slice_prob', label: 'slice prob', ptype: 'float', min: 0, max: 1, step: 0.01, def: 0, automatable: true, channels: 'layer' },
      { key: 'slice_speed', label: 'slice spd', ptype: 'float', min: 0, max: 30, step: 1, def: 0, automatable: true, channels: 'layer' },
      { key: 'slice_axis', label: 'slice axis', ptype: 'enum', def: 0, automatable: false, channels: 'layer',
        options: [ { value: 0, label: 'H' }, { value: 1, label: 'V' }, { value: 2, label: 'Both' } ] },
      { key: 'block_size', label: 'block sz', ptype: 'float', min: 4, max: 128, step: 1, def: 4, automatable: true, channels: 'layer' },
      { key: 'block_intensity', label: 'block int', ptype: 'float', min: 0, max: 1, step: 0.01, def: 0, automatable: true, channels: 'layer' },
      { key: 'block_prob', label: 'block prob', ptype: 'float', min: 0, max: 1, step: 0.01, def: 0, automatable: true, channels: 'layer' },
      { key: 'block_speed', label: 'block spd', ptype: 'float', min: 0, max: 30, step: 1, def: 0, automatable: true, channels: 'layer' },
      { key: 'shift_chroma', label: 'shift chr', ptype: 'float', min: 0, max: 1, step: 0.01, def: 0, automatable: true, channels: 'layer' },
      { key: 'jitter_amount', label: 'jitter amt', ptype: 'float', min: 0, max: 1, step: 0.01, def: 0, automatable: true, channels: 'layer' },
      { key: 'jitter_speed', label: 'jitter spd', ptype: 'float', min: 0, max: 30, step: 1, def: 0, automatable: true, channels: 'layer' },
      { key: 'datamosh', label: 'datamosh', ptype: 'float', min: 0, max: 1, step: 0.01, def: 0, automatable: true, channels: 'layer' },
    ],
  },
  {
    name: 'FEEDBACK',
    params: [
      { key: 'feedback_persistence', label: 'persist', ptype: 'float', min: 0, max: 1, step: 0.01, def: 0, automatable: true, channels: 'layer' },
      { key: 'feedback_zoom', label: 'fb zoom', ptype: 'float', min: 0.8, max: 1.2, step: 0.005, def: 1, automatable: true, channels: 'layer' },
      { key: 'feedback_rotate', label: 'fb rotate', ptype: 'bipolar', min: -30, max: 30, step: 0.5, def: 0, automatable: true, channels: 'layer' },
      { key: 'feedback_luma_key', label: 'fb luma', ptype: 'float', min: 0, max: 1, step: 0.01, def: 0, automatable: true, channels: 'layer' },
      { key: 'feedback_chroma', label: 'fb chroma', ptype: 'float', min: 0, max: 1, step: 0.01, def: 0, automatable: true, channels: 'layer' },
      { key: 'feedback_additive', label: 'fb add', ptype: 'float', min: 0, max: 1, step: 0.01, def: 0, automatable: true, channels: 'layer' },
    ],
  },
  {
    name: 'TRANSFORM',
    params: [
      { key: 'layer_x', label: 'pos x', ptype: 'bipolar', min: -1, max: 1, step: 0.01, def: 0, automatable: true, channels: 'layer' },
      { key: 'layer_y', label: 'pos y', ptype: 'bipolar', min: -1, max: 1, step: 0.01, def: 0, automatable: true, channels: 'layer' },
      { key: 'layer_scale', label: 'scale', ptype: 'float', min: 0.1, max: 4, step: 0.01, def: 1, automatable: true, channels: 'layer' },
      { key: 'fit_mode', label: 'fit', ptype: 'enum', def: 0, automatable: false, channels: 'layer',
        options: [ { value: 0, label: 'Stretch' }, { value: 1, label: 'Fit' }, { value: 2, label: 'Fill' } ] },
    ],
  },
  {
    name: 'VHS/NTSC',
    params: [
      { key: 'enabled', label: 'vhs on', ptype: 'bool', def: false, automatable: false, channels: 'ntsc' },
      { key: 'tape_speed', label: 'tape', ptype: 'enum', def: 0, automatable: false, channels: 'ntsc',
        options: [ { value: 0, label: 'SP' }, { value: 1, label: 'LP' }, { value: 2, label: 'EP' } ] },
      { key: 'chroma_loss', label: 'chroma loss', ptype: 'float', min: 0, max: 0.01, step: 0.0005, def: 0, automatable: false, channels: 'ntsc' },
      { key: 'edge_wave_enabled', label: 'edge on', ptype: 'bool', def: false, automatable: false, channels: 'ntsc' },
      { key: 'edge_wave_intensity', label: 'edge int', ptype: 'float', min: 0, max: 20, step: 0.5, def: 0, automatable: false, channels: 'ntsc' },
      { key: 'edge_wave_speed', label: 'edge spd', ptype: 'float', min: 0, max: 10, step: 0.1, def: 0.5, automatable: false, channels: 'ntsc' },
      { key: 'head_switching_enabled', label: 'head on', ptype: 'bool', def: false, automatable: false, channels: 'ntsc' },
      { key: 'head_switching_height', label: 'head ht', ptype: 'float', min: 0, max: 24, step: 1, def: 8, automatable: false, channels: 'ntsc' },
      { key: 'head_switching_shift', label: 'head shift', ptype: 'bipolar', min: -100, max: 100, step: 1, def: 0, automatable: false, channels: 'ntsc' },
      { key: 'tracking_noise_enabled', label: 'track on', ptype: 'bool', def: false, automatable: false, channels: 'ntsc' },
      { key: 'tracking_noise_height', label: 'track ht', ptype: 'float', min: 0, max: 120, step: 1, def: 24, automatable: false, channels: 'ntsc' },
      { key: 'tracking_noise_wave', label: 'track wave', ptype: 'float', min: 0, max: 50, step: 0.5, def: 0, automatable: false, channels: 'ntsc' },
      { key: 'tracking_noise_snow', label: 'track snow', ptype: 'float', min: 0, max: 1, step: 0.01, def: 0, automatable: false, channels: 'ntsc' },
      { key: 'snow_intensity', label: 'snow', ptype: 'float', min: 0, max: 1, step: 0.01, def: 0, automatable: false, channels: 'ntsc' },
      { key: 'composite_noise_intensity', label: 'comp noise', ptype: 'float', min: 0, max: 0.5, step: 0.005, def: 0, automatable: false, channels: 'ntsc' },
      { key: 'luma_noise_intensity', label: 'luma noise', ptype: 'float', min: 0, max: 0.2, step: 0.005, def: 0, automatable: false, channels: 'ntsc' },
      { key: 'chroma_noise_intensity', label: 'chr noise', ptype: 'float', min: 0, max: 0.5, step: 0.005, def: 0, automatable: false, channels: 'ntsc' },
      { key: 'luma_smear', label: 'luma smear', ptype: 'float', min: 0, max: 1, step: 0.01, def: 0, automatable: false, channels: 'ntsc' },
      { key: 'composite_sharpening', label: 'sharpen', ptype: 'bipolar', min: -1, max: 2, step: 0.05, def: 0, automatable: false, channels: 'ntsc' },
    ],
  },
];

// Does a param row apply to a given column kind?
//   'both' → master + layer columns.
//   'ntsc' → master column (VHS is a global effect, shown under Master — mirrors
//            the classic panel where #vhs-group lives inside #master-fx).
//   otherwise an exact match.
function CHANNEL_APPLIES(def, colKind) {
  if (def.channels === 'both') return colKind === 'master' || colKind === 'layer';
  if (def.channels === 'ntsc') return colKind === 'master';
  return def.channels === colKind;
}

window.MATRIX_GROUPS = MATRIX_GROUPS;
window.CHANNEL_APPLIES = CHANNEL_APPLIES;

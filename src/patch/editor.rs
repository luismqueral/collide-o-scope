//! In-window YAML/parameter editor (egui).
//!
//! This is the small Ctrl+E editor panel drawn over the render window — a
//! fallback / power-user view; most control happens in the browser. It's built
//! with egui, an *immediate-mode* GUI: there are no persistent widget objects.
//! Every frame we re-run the build functions, which describe the UI from current
//! state and return what the user did this frame (clicks, edits). Persistent bits
//! (which field is being edited, its text buffer) therefore live in `EditorState`,
//! not in the widgets themselves.

use std::collections::HashMap;

use crate::automation::Expr;
use crate::effects::EffectUniforms;
use crate::layers::Layer;
use crate::ntsc::NtscParams;

use super::{param_meta, EffectsConfig, LayerConfig, PatchState};

/// State that must survive between frames (egui widgets don't hold their own).
pub struct EditorState {
    pub active: bool,
    pub tab: usize,              // 0 = Master, 1+ = layer index + 1
    pub active_field: Option<String>, // which param key is being edited
    pub field_buffer: String,    // text in the active pill input
    pub request_focus: bool,     // request focus on next frame
}

impl Default for EditorState {
    fn default() -> Self {
        Self {
            active: true,
            tab: 0,
            active_field: None,
            field_buffer: String::new(),
            request_focus: false,
        }
    }
}

// Colors for syntax-highlighted code look
const KEY_COLOR: egui::Color32 = egui::Color32::from_rgb(130, 170, 255);
const VALUE_COLOR: egui::Color32 = egui::Color32::from_rgb(200, 230, 200);
const BOOL_COLOR: egui::Color32 = egui::Color32::from_rgb(255, 180, 100);
const STRING_COLOR: egui::Color32 = egui::Color32::from_rgb(200, 160, 255);
const PILL_BG: egui::Color32 = egui::Color32::from_rgb(50, 55, 65);
const GROUP_COLOR: egui::Color32 = egui::Color32::from_rgb(90, 130, 160);

// Key column width in pixels (enough for "breathe_rotation:" in monospace 13)
const KEY_COL_WIDTH: f32 = 140.0;

/// Render a single param row: key: [value pill]
/// `key_width` aligns the value column (in pixels). Pass 0.0 for no alignment.
/// Returns the new value if changed this frame.
///
/// Immediate-mode flow: this runs every frame. The row has two visual states —
/// an inactive clickable "pill" (just a label) and, once clicked, an active
/// text input. Which one is showing is decided purely by comparing `key` against
/// `editor.active_field` (the persistent bit), since the widgets themselves keep
/// no state between frames. We return `Some(text)` if the user changed the value
/// this frame, `None` otherwise — the caller decides what to do with it.
fn param_row(
    ui: &mut egui::Ui,
    key: &str,
    value: &str,
    editor: &mut EditorState,
    key_width: f32,
) -> Option<String> {
    let mut new_value: Option<String> = None;
    // Look up step/min/max/description for this key (None for unknown keys).
    let meta = param_meta(key);

    // Fixed pill width for consistent alignment
    const PILL_WIDTH: f32 = 72.0;

    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 2.0;

        // Key label (padded to key_width for alignment)
        let key_text = format!("{}:", key);
        if key_width > 0.0 {
            ui.allocate_ui_with_layout(
                egui::vec2(key_width, ui.spacing().interact_size.y),
                egui::Layout::left_to_right(egui::Align::Center),
                |ui| {
                    ui.label(
                        egui::RichText::new(&key_text)
                            .monospace()
                            .size(13.0)
                            .color(KEY_COLOR),
                    );
                },
            );
        } else {
            ui.label(
                egui::RichText::new(&key_text)
                    .monospace()
                    .size(13.0)
                    .color(KEY_COLOR),
            );
        }

        // This row is "active" (editable) only if it's the one field the editor
        // currently has open. as_deref() turns Option<String> into Option<&str>
        // so it can be compared against Some(key) without allocating.
        let is_active = editor.active_field.as_deref() == Some(key);

        let pill_response = if is_active {
            // Active pill: singleline text input (fixed width). egui re-creates
            // this widget every frame; the text it shows/edits lives in
            // editor.field_buffer, not in the widget. make_persistent_id gives it
            // a stable id across frames so egui can track focus/cursor for it.
            let id = ui.make_persistent_id(format!("pill_{}", key));
            let response = ui.add(
                egui::TextEdit::singleline(&mut editor.field_buffer)
                    .id(id)
                    .desired_width(PILL_WIDTH)
                    .font(egui::FontId::monospace(13.0))
                    .background_color(PILL_BG),
            );

            // Request focus on activation frame. The frame the user clicks the
            // pill we set request_focus=true; here (the next frame, when the input
            // actually exists) we hand it keyboard focus and select all its text,
            // then clear the flag so this only happens once.
            if editor.request_focus {
                response.request_focus();
                // Select all text so typing replaces the old value immediately.
                if let Some(mut state) = egui::TextEdit::load_state(ui.ctx(), id) {
                    state.cursor.set_char_range(Some(egui::text::CCursorRange::two(
                        egui::text::CCursor::new(0),
                        egui::text::CCursor::new(editor.field_buffer.len()),
                    )));
                    state.store(ui.ctx(), id);
                }
                editor.request_focus = false;
            }

            // Up/Down arrow stepping: nudge the numeric value by meta.step,
            // clamped to [min, max]. Only works when we have meta (known param)
            // and the current buffer parses as a number; otherwise it no-ops.
            let up = ui.input(|i| i.key_pressed(egui::Key::ArrowUp));
            let down = ui.input(|i| i.key_pressed(egui::Key::ArrowDown));
            if (up || down) && meta.is_some() {
                let m = meta.as_ref().unwrap();
                if let Ok(current) = editor.field_buffer.parse::<f32>() {
                    let delta = if up { m.step } else { -m.step };
                    let stepped = (current + delta).clamp(m.min, m.max);
                    editor.field_buffer = format_value(stepped, m.step);
                    new_value = Some(editor.field_buffer.clone());
                }
            }

            // Apply on every text change (live update as you type).
            // response.changed() is egui's "the contents differ from last frame".
            if response.changed() {
                new_value = Some(editor.field_buffer.clone());
            }

            // Confirm and close the editor on Enter or when focus is lost
            // (e.g. clicking elsewhere). lost_focus() fires the frame focus leaves.
            let enter = ui.input(|i| i.key_pressed(egui::Key::Enter));
            if enter || response.lost_focus() {
                new_value = Some(editor.field_buffer.clone());
                editor.active_field = None;
            }

            // Escape closes the editor WITHOUT emitting a new value (cancel).
            if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                editor.active_field = None;
            }

            response
        } else {
            // Inactive: a clickable value label styled to look like a pill.
            // Colour-code by inferred type so the editor reads like syntax-
            // highlighted YAML: orange bools, green numbers, purple strings.
            let color = if value == "true" || value == "false" {
                BOOL_COLOR
            } else if value.parse::<f64>().is_ok() {
                VALUE_COLOR
            } else {
                STRING_COLOR
            };

            let inner = ui.allocate_ui_with_layout(
                egui::vec2(PILL_WIDTH, ui.spacing().interact_size.y),
                egui::Layout::left_to_right(egui::Align::Center),
                |ui| {
                    ui.add(
                        egui::Label::new(
                            egui::RichText::new(value)
                                .monospace()
                                .size(13.0)
                                .color(color)
                                .background_color(PILL_BG),
                        )
                        .sense(egui::Sense::click()),
                    )
                },
            );

            // Clicking the pill switches this row into edit mode next frame:
            // record which field is active, seed the text buffer with the current
            // value, and ask for focus (handled at the top of the active branch).
            let response = inner.inner;
            if response.clicked() {
                editor.active_field = Some(key.to_string());
                editor.field_buffer = value.to_string();
                editor.request_focus = true;
            }

            response
        };

        // Show the param's description + range as a hover tooltip (keeps the row
        // compact — the help text only appears on mouse-over).
        if let Some(m) = &meta {
            pill_response.on_hover_text(format!("{} [{} to {}]", m.desc, m.min, m.max));
        }
    });

    new_value
}

/// Format a stepped value with a sensible number of decimals: coarse steps
/// (>=1) show one decimal, medium steps two, fine steps three. This keeps the
/// pill from showing noise like "0.30000001" after arrow-key stepping.
fn format_value(v: f32, step: f32) -> String {
    if step >= 1.0 {
        format!("{:.1}", v)
    } else if step >= 0.01 {
        format!("{:.2}", v)
    } else {
        format!("{:.3}", v)
    }
}

/// Build the YAML editor content (rendered inside the shared left panel).
///
/// Top-level layout of one frame: a tab bar (Master + one tab per layer), then
/// the fields for whichever tab is selected. Edits flow through a small
/// "config" snapshot: we read the live state into a Config, let param_row mutate
/// a clone of it, and only write the clone back to the live struct if something
/// actually changed this frame. That avoids touching GPU/runtime state on the
/// vast majority of frames where the user isn't editing anything.
pub fn build_yaml_editor_content(
    ui: &mut egui::Ui,
    layers: &mut Vec<Layer>,
    master_effects: &mut EffectUniforms,
    editor: &mut EditorState,
) {
    // Tab bar: Master + per-layer tabs. selectable_label draws a toggle-styled
    // button; switching tabs also clears any in-progress field edit.
    ui.horizontal(|ui| {
        if ui
            .selectable_label(editor.tab == 0, "Master")
            .clicked()
            && editor.tab != 0
        {
            editor.tab = 0;
            editor.active_field = None;
        }

        for i in 0..layers.len() {
            let tab_id = i + 1;
            let label = format!("Layer {}", i + 1);
            if ui
                .selectable_label(editor.tab == tab_id, &label)
                .clicked()
                && editor.tab != tab_id
            {
                editor.tab = tab_id;
                editor.active_field = None;
            }
        }
    });

    ui.separator();

    // Render fields based on active tab
    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            // tab 0 = Master output bus; tab n = layer (n-1).
            match editor.tab {
                0 => {
                    // Master effects — snapshot uniforms into a config, render its
                    // fields grouped by category, and collect edits into a clone.
                    let config = EffectsConfig::from_uniforms(master_effects);
                    let groups = config.grouped_fields();
                    let mut updated_config = config.clone();
                    let mut changed = false;

                    for (group_name, fields) in &groups {
                        ui.add_space(6.0);
                        ui.label(
                            egui::RichText::new(format!("# {}", group_name))
                                .monospace()
                                .size(11.0)
                                .color(GROUP_COLOR),
                        );
                        for (key, value) in fields {
                            if let Some(new_val) = param_row(ui, key, value, editor, KEY_COL_WIDTH) {
                                if updated_config.set_field(key, &new_val) {
                                    changed = true;
                                }
                            }
                        }
                    }

                    // Only write back to the live uniforms if a field changed.
                    if changed {
                        updated_config.apply_to_uniforms(master_effects);
                    }
                }
                n => {
                    // Per-layer tab. Guard idx in case layers changed since the
                    // tab was selected (e.g. a layer was removed).
                    let idx = n - 1;
                    if idx < layers.len() {
                        let config = LayerConfig::from_layer(&layers[idx]);
                        let top_fields = config.top_fields();
                        let effect_groups = config.effects.grouped_fields();

                        let mut updated_config = config.clone();
                        let mut changed = false;

                        // Layer top-level fields (filename, blend, opacity, …).
                        for (key, value) in &top_fields {
                            if *key == "filename" {
                                // Read-only filename: rendered as a plain label
                                // (no clickable pill) since you can't retype the
                                // source clip from here.
                                ui.horizontal(|ui| {
                                    ui.spacing_mut().item_spacing.x = 2.0;
                                    ui.allocate_ui_with_layout(
                                        egui::vec2(KEY_COL_WIDTH, ui.spacing().interact_size.y),
                                        egui::Layout::left_to_right(egui::Align::Center),
                                        |ui| {
                                            ui.label(
                                                egui::RichText::new("filename:")
                                                    .monospace()
                                                    .size(13.0)
                                                    .color(KEY_COLOR),
                                            );
                                        },
                                    );
                                    ui.label(
                                        egui::RichText::new(value)
                                            .monospace()
                                            .size(13.0)
                                            .color(STRING_COLOR),
                                    );
                                });
                                continue;
                            }
                            if let Some(new_val) = param_row(ui, key, value, editor, KEY_COL_WIDTH) {
                                if updated_config.set_field(key, &new_val) {
                                    changed = true;
                                }
                            }
                        }

                        ui.add_space(8.0);
                        ui.label(
                            egui::RichText::new("effects:")
                                .monospace()
                                .size(13.0)
                                .color(KEY_COLOR),
                        );

                        // Effect fields — grouped
                        for (group_name, fields) in &effect_groups {
                            ui.add_space(6.0);
                            ui.label(
                                egui::RichText::new(format!("  # {}", group_name))
                                    .monospace()
                                    .size(11.0)
                                    .color(GROUP_COLOR),
                            );
                            for (key, value) in fields {
                                if let Some(new_val) = param_row(ui, key, value, editor, KEY_COL_WIDTH) {
                                    if updated_config.effects.set_field(key, &new_val) {
                                        changed = true;
                                    }
                                }
                            }
                        }

                        // Single write-back for this layer if anything changed.
                        if changed {
                            updated_config.apply_to_layer(&mut layers[idx]);
                        }
                    }
                }
            }
        });

    // Status line — a quiet (weak = dimmed) hint at the bottom of the panel.
    ui.add_space(4.0);
    ui.separator();
    ui.weak("click value to edit · ↑↓ step · enter to confirm");
}

/// Save full patch state to a YAML file via native dialog.
///
/// Three steps: snapshot the live engine state into a serializable PatchState,
/// serialise it to YAML text, then ask the OS for a save path via rfd (a native
/// file dialog). The dialog returns None if the user cancels, so the whole
/// write is wrapped in `if let Some(path)` — cancel = do nothing.
pub fn save_patch(
    master: &EffectUniforms,
    master_automations: &HashMap<String, Expr>,
    layers: &[Layer],
    ntsc_params: &NtscParams,
) {
    let patch = PatchState::capture(master, master_automations, layers, ntsc_params);
    // unwrap_or_default: on the (unexpected) serialise error, fall back to ""
    // rather than panicking — the user just gets an empty/failed save.
    let yaml = serde_yaml::to_string(&patch).unwrap_or_default();

    if let Some(path) = rfd::FileDialog::new()
        .set_file_name("patch.yaml")
        .add_filter("YAML", &["yaml", "yml"])
        .save_file()
    {
        // Errors are reported to stderr only — this is a power-user fallback,
        // so we don't surface a GUI error dialog.
        if let Err(e) = std::fs::write(&path, &yaml) {
            eprintln!("Failed to save patch: {e}");
        }
    }
}

/// Load a patch from a YAML file via native dialog.
///
/// Returns the recompiled master automations (and any parse errors) so the
/// caller can install them on the app; `None` if the dialog was cancelled or
/// the file failed to load.
pub fn load_patch(
    master: &mut EffectUniforms,
    layers: &mut Vec<Layer>,
    ntsc_params: &mut NtscParams,
) -> Option<(HashMap<String, Expr>, HashMap<String, String>)> {
    // pick_file() returns Option<PathBuf>; the `?` early-returns None from this
    // whole function if the user cancels the dialog.
    let path = rfd::FileDialog::new()
        .add_filter("YAML", &["yaml", "yml"])
        .pick_file()?;

    // Two fallible steps, each handled with match: read the file, then parse it.
    // Any failure logs to stderr and returns None so the caller leaves the
    // current patch untouched.
    match std::fs::read_to_string(&path) {
        Ok(yaml) => match serde_yaml::from_str::<PatchState>(&yaml) {
            Ok(patch) => {
                // apply() writes the loaded settings onto the live state in place;
                // compile_master_automations() turns the saved expression strings
                // back into runnable Exprs for the caller to install.
                patch.apply(master, layers, ntsc_params);
                Some(patch.compile_master_automations())
            }
            Err(e) => {
                eprintln!("Failed to parse patch: {e}");
                None
            }
        },
        Err(e) => {
            eprintln!("Failed to read file: {e}");
            None
        }
    }
}

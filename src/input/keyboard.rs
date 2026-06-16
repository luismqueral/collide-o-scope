//! In-window keyboard shortcuts.
//!
//! This covers only the quick render-window shortcuts (pixelate, RGB split,
//! reset, pause, fullscreen, quit). Other keys (Ctrl+E/S/O for the YAML editor
//! and patch save/load) are handled directly in `main.rs`'s event loop, not
//! here. The flow is: `map_key` turns a raw key event into an `Action`, then
//! `apply_action` performs it and returns a `ControlFlow` telling the caller
//! whether it needs to do something window-level (pause/fullscreen/quit).
use winit::event::ElementState;
use winit::keyboard::{KeyCode, PhysicalKey};

use crate::effects::EffectUniforms;

/// Actions that can be triggered by keyboard input. `None` is the explicit
/// "key we don't handle" case — Rust enums must cover every possibility, so
/// there's no implicit null.
pub enum Action {
    IncreasePixelate,
    DecreasePixelate,
    IncreaseRgbSplit,
    DecreaseRgbSplit,
    ResetEffects,
    TogglePause,
    ToggleFullscreen,
    Quit,
    None,
}

/// Map a physical key press to an action. Shift modifies direction.
///
/// We only act on key-*down* (`Pressed`); key-up events return `None` so an
/// action doesn't fire twice per physical press. "Physical" key means the
/// position on the keyboard regardless of layout/locale.
pub fn map_key(key: PhysicalKey, state: ElementState, shift: bool) -> Action {
    if state != ElementState::Pressed {
        return Action::None;
    }

    // `match` is Rust's exhaustive switch. Each `=>` arm returns an `Action`;
    // the final `_` arm catches every other key.
    match key {
        PhysicalKey::Code(KeyCode::KeyP) => {
            if shift {
                Action::DecreasePixelate
            } else {
                Action::IncreasePixelate
            }
        }
        PhysicalKey::Code(KeyCode::KeyG) => {
            if shift {
                Action::DecreaseRgbSplit
            } else {
                Action::IncreaseRgbSplit
            }
        }
        PhysicalKey::Code(KeyCode::Digit0) => Action::ResetEffects,
        PhysicalKey::Code(KeyCode::Space) => Action::TogglePause,
        PhysicalKey::Code(KeyCode::KeyF) => Action::ToggleFullscreen,
        PhysicalKey::Code(KeyCode::Escape) => Action::Quit,
        _ => Action::None,
    }
}

/// Apply an action to effect uniforms. Returns a `ControlFlow` so the caller
/// (the winit event loop) can react to window-level actions it owns — uniforms
/// are mutated here in place (`&mut`), but pausing/fullscreen/quit need the
/// window, which lives in `main.rs`.
pub fn apply_action(action: Action, uniforms: &mut EffectUniforms) -> ControlFlow {
    match action {
        Action::IncreasePixelate => uniforms.increase_pixelate(),
        Action::DecreasePixelate => uniforms.decrease_pixelate(),
        Action::IncreaseRgbSplit => uniforms.increase_rgb_split(),
        Action::DecreaseRgbSplit => uniforms.decrease_rgb_split(),
        Action::ResetEffects => uniforms.reset(),
        Action::TogglePause => return ControlFlow::TogglePause,
        Action::ToggleFullscreen => return ControlFlow::ToggleFullscreen,
        Action::Quit => return ControlFlow::Quit,
        Action::None => {}
    }
    ControlFlow::Continue
}

/// What the caller should do after an action runs. `Continue` means "nothing
/// special, keep going"; the others ask the event loop to touch window state.
pub enum ControlFlow {
    Continue,
    TogglePause,
    ToggleFullscreen,
    Quit,
}

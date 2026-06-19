//! Shared test-only helpers for synthesizing tiny media fixtures with the
//! system `ffmpeg` CLI (already a hard dependency of this project).
//!
//! Each fixture lives in its own [`tempfile::TempDir`], which deletes itself
//! when dropped — so callers must keep the returned `TempDir` in scope for as
//! long as the file is needed. Nothing here is committed to git; the clips are
//! regenerated at test time.

use std::path::PathBuf;
use std::process::Command;
use tempfile::TempDir;

/// Run an `ffmpeg` command, panicking with its stderr on failure so a broken
/// fixture surfaces as a clear error rather than a confusing decode failure.
fn run_ffmpeg(args: &[&str]) {
    let output = Command::new("ffmpeg")
        .args(args)
        .output()
        .expect("failed to spawn `ffmpeg` (is it installed and on PATH?)");
    assert!(
        output.status.success(),
        "ffmpeg failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

/// Synthesize a tiny silent test video (`testsrc`, video-only — no audio track)
/// into a fresh tempdir. Returns `(dir, path)`; keep `dir` alive.
pub fn synth_video(width: u32, height: u32, fps: u32, secs: f32) -> (TempDir, PathBuf) {
    let dir = tempfile::tempdir().expect("create tempdir");
    let path = dir.path().join("clip.mp4");
    run_ffmpeg(&[
        "-v",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        &format!("testsrc=duration={secs}:size={width}x{height}:rate={fps}"),
        "-pix_fmt",
        "yuv420p",
        path.to_str().unwrap(),
    ]);
    (dir, path)
}

/// Synthesize a tiny audio-only file (`sine` tone) into a fresh tempdir.
/// Returns `(dir, path)`; keep `dir` alive.
pub fn synth_audio(freq: u32, secs: f32) -> (TempDir, PathBuf) {
    let dir = tempfile::tempdir().expect("create tempdir");
    let path = dir.path().join("tone.m4a");
    run_ffmpeg(&[
        "-v",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        &format!("sine=frequency={freq}:duration={secs}"),
        path.to_str().unwrap(),
    ]);
    (dir, path)
}

/// Write a plain-text (non-media) file into a fresh tempdir. Used to assert the
/// decoders reject files that contain no decodable stream. Returns `(dir, path)`.
pub fn write_non_media() -> (TempDir, PathBuf) {
    let dir = tempfile::tempdir().expect("create tempdir");
    let path = dir.path().join("not-media.txt");
    std::fs::write(&path, b"this is plainly not a media file").expect("write text file");
    (dir, path)
}

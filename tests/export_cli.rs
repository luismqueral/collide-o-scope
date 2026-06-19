//! Tier 3 — headless GPU export end-to-end test.
//!
//! Drives the *real* built binary's `render` subcommand as a subprocess, which
//! exercises the whole offline path: CLI parse → headless wgpu device → ffmpeg
//! decode → per-layer + master FX → composite → MP4 encode. This is the only
//! `tests/` file in the project: a binary crate exposes no library API to a
//! `tests/` integration crate, but it *can* spawn the compiled binary via the
//! `CARGO_BIN_EXE_<name>` env var Cargo provides to integration tests.
//!
//! Marked `#[ignore]` because GitHub's hosted runners have no reliable GPU;
//! run locally with `cargo nextest run --run-ignored all` (Mac/Metal).

use std::path::Path;
use std::process::Command;

/// Synthesize a tiny silent test clip with the system ffmpeg CLI.
fn synth_clip(path: &Path) {
    let status = Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=0.5:size=64x48:rate=10",
            "-pix_fmt",
            "yuv420p",
            path.to_str().unwrap(),
        ])
        .status()
        .expect("failed to spawn ffmpeg to build the fixture");
    assert!(status.success(), "ffmpeg fixture synthesis failed");
}

/// End-to-end: the built binary's `render` subcommand decodes a clip, runs the
/// full FX/composite path, and writes a non-trivial MP4 to the requested path.
#[test]
#[ignore = "requires GPU (headless wgpu) — run with --run-ignored all"]
fn render_subcommand_exports_an_mp4() {
    eprintln!("export-e2e: render subcommand produces a non-trivial MP4 end-to-end");
    let dir = tempfile::tempdir().expect("create tempdir");
    let lib = dir.path();

    // Library: a single synthesized clip.
    let clip = lib.join("clip.mp4");
    synth_clip(&clip);

    // Minimal patch referencing the clip by bare filename (the exporter joins
    // it onto the library folder). All effect fields rely on serde defaults.
    let patch_path = lib.join("patch.yaml");
    std::fs::write(&patch_path, "master: {}\nlayers:\n  - filename: clip.mp4\n")
        .expect("write patch yaml");

    let out = lib.join("out.mp4");

    let status = Command::new(env!("CARGO_BIN_EXE_collide-o-scope"))
        .args([
            "render",
            "--patch",
            patch_path.to_str().unwrap(),
            "--library",
            lib.to_str().unwrap(),
            "--out",
            out.to_str().unwrap(),
            "--duration",
            "0.5",
            "--fps",
            "10",
            "--res",
            "128x96",
        ])
        .status()
        .expect("failed to spawn the collide-o-scope binary");

    assert!(status.success(), "render subcommand exited non-zero");
    assert!(out.exists(), "expected output MP4 was not created");
    let size = std::fs::metadata(&out).expect("stat output").len();
    // A real encoded MP4 is at least a few KB; guard against an empty/truncated
    // file that would still "exist".
    assert!(size > 1_000, "output MP4 suspiciously small: {size} bytes");
}

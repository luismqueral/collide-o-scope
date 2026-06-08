#![allow(deprecated)] // egui 0.34 deprecation warnings for panel API renames

mod effects;
mod input;
mod layers;
mod renderer;
mod video;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use egui_wgpu::ScreenDescriptor;
use winit::application::ApplicationHandler;
use winit::event::{KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::ModifiersState;
use winit::window::{Fullscreen, Window, WindowAttributes, WindowId};

use input::{apply_action, map_key, ControlFlow};
use layers::{is_video_file, BlendMode, Layer};
use renderer::Renderer;

const TARGET_FPS: u64 = 30;
const FRAME_DURATION: Duration = Duration::from_millis(1000 / TARGET_FPS);

struct App {
    initial_video: Option<String>,
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    layers: Vec<Layer>,
    selected_layer: Option<usize>,
    master_effects: effects::EffectUniforms,
    master_paused: bool,
    last_frame_time: Instant,
    start_time: Instant,
    modifiers: ModifiersState,
    // Library
    library_folder: Option<PathBuf>,
    library_files: Vec<PathBuf>,
    // egui state
    egui_ctx: egui::Context,
    egui_winit: Option<egui_winit::State>,
    egui_renderer: Option<egui_wgpu::Renderer>,
    video_egui_texture_id: Option<egui::TextureId>,
}

impl App {
    fn new(initial_video: Option<String>, library_folder: Option<PathBuf>) -> Self {
        let library_files = library_folder
            .as_ref()
            .map(|f| scan_folder(f))
            .unwrap_or_default();

        Self {
            initial_video,
            window: None,
            renderer: None,
            layers: Vec::new(),
            selected_layer: None,
            master_effects: effects::EffectUniforms::default(),
            master_paused: false,
            last_frame_time: Instant::now(),
            start_time: Instant::now(),
            modifiers: ModifiersState::empty(),
            library_folder,
            library_files,
            egui_ctx: egui::Context::default(),
            egui_winit: None,
            egui_renderer: None,
            video_egui_texture_id: None,
        }
    }

    fn add_layer(&mut self, path: &str) {
        let renderer = self.renderer.as_ref().unwrap();
        match Layer::new(path, &renderer.device) {
            Ok(layer) => {
                self.layers.push(layer);
                self.selected_layer = Some(self.layers.len() - 1);
            }
            Err(e) => {
                eprintln!("Failed to open video: {e}");
            }
        }
    }

    fn set_library_folder(&mut self, folder: PathBuf) {
        self.library_files = scan_folder(&folder);
        self.library_folder = Some(folder);
    }
}

/// Scan a directory for video files, returning sorted list of paths.
fn scan_folder(folder: &PathBuf) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(folder) else {
        return Vec::new();
    };
    let mut files: Vec<PathBuf> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_file() && is_video_file(p))
        .collect();
    files.sort();
    files
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let mut output_width = 1280u32;
        let mut output_height = 720u32;

        if let Some(ref path) = self.initial_video {
            if let Ok(decoder) = video::VideoDecoder::open(path) {
                output_width = decoder.width;
                output_height = decoder.height;
            }
        }

        log::info!("Output: {}x{}", output_width, output_height);

        let window_w = output_width + 460;
        let window_h = output_height;

        let window_attrs = WindowAttributes::default()
            .with_title("collide-o-scope")
            .with_inner_size(winit::dpi::LogicalSize::new(window_w, window_h));

        let window = Arc::new(event_loop.create_window(window_attrs).unwrap());

        let renderer = Renderer::new(window.clone(), output_width, output_height);

        configure_fonts(&self.egui_ctx);

        let egui_winit = egui_winit::State::new(
            self.egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            None,
            None,
        );

        let mut egui_renderer = egui_wgpu::Renderer::new(
            &renderer.device,
            renderer.config.format,
            egui_wgpu::RendererOptions::default(),
        );

        let video_egui_texture_id = egui_renderer.register_native_texture(
            &renderer.device,
            &renderer.output_view,
            wgpu::FilterMode::Linear,
        );

        self.window = Some(window);
        self.renderer = Some(renderer);
        self.egui_winit = Some(egui_winit);
        self.egui_renderer = Some(egui_renderer);
        self.video_egui_texture_id = Some(video_egui_texture_id);

        if let Some(path) = self.initial_video.take() {
            self.add_layer(&path);
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        if let Some(egui_winit) = &mut self.egui_winit {
            let response =
                egui_winit.on_window_event(self.window.as_ref().unwrap(), &event);
            if response.consumed {
                return;
            }
        }

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }

            WindowEvent::Resized(new_size) => {
                if let Some(renderer) = &mut self.renderer {
                    renderer.resize(new_size.width, new_size.height);
                }
            }

            WindowEvent::ScaleFactorChanged { .. } => {
                let size = self.window.as_ref().unwrap().inner_size();
                if let Some(renderer) = &mut self.renderer {
                    renderer.resize(size.width, size.height);
                }
            }

            WindowEvent::ModifiersChanged(mods) => {
                self.modifiers = mods.state();
            }

            WindowEvent::DroppedFile(path) => {
                if path.is_dir() {
                    self.set_library_folder(path);
                } else if is_video_file(&path) {
                    if let Some(path_str) = path.to_str() {
                        let path_owned = path_str.to_string();
                        self.add_layer(&path_owned);
                    }
                }
            }

            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key,
                        state,
                        ..
                    },
                ..
            } => {
                let shift = self.modifiers.shift_key();
                let action = map_key(physical_key, state, shift);

                if let Some(idx) = self.selected_layer {
                    if let Some(layer) = self.layers.get_mut(idx) {
                        match apply_action(action, &mut layer.effects) {
                            ControlFlow::Quit => event_loop.exit(),
                            ControlFlow::TogglePause => layer.paused = !layer.paused,
                            ControlFlow::ToggleFullscreen => {
                                if let Some(window) = &self.window {
                                    let current = window.fullscreen();
                                    if current.is_some() {
                                        window.set_fullscreen(None);
                                    } else {
                                        window.set_fullscreen(Some(Fullscreen::Borderless(None)));
                                    }
                                }
                            }
                            ControlFlow::Continue => {}
                        }
                    }
                } else {
                    let mut dummy = effects::EffectUniforms::default();
                    match apply_action(action, &mut dummy) {
                        ControlFlow::Quit => event_loop.exit(),
                        ControlFlow::ToggleFullscreen => {
                            if let Some(window) = &self.window {
                                let current = window.fullscreen();
                                if current.is_some() {
                                    window.set_fullscreen(None);
                                } else {
                                    window.set_fullscreen(Some(Fullscreen::Borderless(None)));
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }

            WindowEvent::RedrawRequested => {
                let win_size = self.window.as_ref().unwrap().inner_size();
                if win_size.width == 0 || win_size.height == 0 {
                    if let Some(w) = &self.window {
                        w.request_redraw();
                    }
                    return;
                }

                let now = Instant::now();
                if now - self.last_frame_time >= FRAME_DURATION {
                    self.last_frame_time = now;

                    // Decode next frame for each layer (per-layer timing, speed, pause)
                    if !self.master_paused {
                        for layer in &mut self.layers {
                            if !layer.paused && layer.ready_for_frame() {
                                if let Some(frame_data) = layer.decoder.next_frame() {
                                    let renderer = self.renderer.as_ref().unwrap();
                                    layer.upload_frame(&renderer.queue, &frame_data);
                                }
                                layer.last_decode = Instant::now();
                            }
                        }
                    }

                    // Build egui UI
                    let window = self.window.as_ref().unwrap();
                    let egui_winit = self.egui_winit.as_mut().unwrap();
                    let raw_input = egui_winit.take_egui_input(window);

                    let layers = &mut self.layers;
                    let selected_layer = &mut self.selected_layer;
                    let master_effects = &mut self.master_effects;
                    let master_paused = &mut self.master_paused;
                    let library_folder = &mut self.library_folder;
                    let library_files = &mut self.library_files;
                    let video_egui_texture_id = self.video_egui_texture_id;
                    let output_width = self.renderer.as_ref().unwrap().output_width;
                    let output_height = self.renderer.as_ref().unwrap().output_height;
                    let mut pending_add: Option<String> = None;

                    let full_output = self.egui_ctx.run_ui(raw_input, |ctx| {
                        pending_add = build_ui(
                            ctx,
                            layers,
                            selected_layer,
                            master_effects,
                            master_paused,
                            library_folder,
                            library_files,
                            video_egui_texture_id,
                            output_width,
                            output_height,
                        );
                    });

                    // Handle deferred layer add (needs device access)
                    if let Some(path) = pending_add {
                        self.add_layer(&path);
                    }

                    let window = self.window.as_ref().unwrap();
                    let egui_winit = self.egui_winit.as_mut().unwrap();
                    egui_winit.handle_platform_output(window, full_output.platform_output);

                    let tris = self
                        .egui_ctx
                        .tessellate(full_output.shapes, full_output.pixels_per_point);

                    // Set time uniform on all effects (drives animated noise/breathing)
                    let elapsed = self.start_time.elapsed().as_secs_f32();
                    for layer in &mut self.layers {
                        layer.effects.time = elapsed;
                    }
                    self.master_effects.time = elapsed;

                    let renderer = self.renderer.as_ref().unwrap();
                    let mut encoder = renderer.device.create_command_encoder(
                        &wgpu::CommandEncoderDescriptor {
                            label: Some("Frame Encoder"),
                        },
                    );
                    renderer.render_layers(&mut encoder, &self.layers);
                    renderer.render_master_effects(&mut encoder, &self.master_effects);

                    let egui_renderer = self.egui_renderer.as_mut().unwrap();
                    for (id, image_delta) in &full_output.textures_delta.set {
                        egui_renderer.update_texture(
                            &renderer.device,
                            &renderer.queue,
                            *id,
                            image_delta,
                        );
                    }

                    let screen_desc = ScreenDescriptor {
                        size_in_pixels: [renderer.config.width, renderer.config.height],
                        pixels_per_point: full_output.pixels_per_point,
                    };

                    egui_renderer.update_buffers(
                        &renderer.device,
                        &renderer.queue,
                        &mut encoder,
                        &tris,
                        &screen_desc,
                    );

                    let surface_texture = match renderer.surface.get_current_texture() {
                        wgpu::CurrentSurfaceTexture::Success(t)
                        | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
                        wgpu::CurrentSurfaceTexture::Outdated
                        | wgpu::CurrentSurfaceTexture::Lost => {
                            let size = window.inner_size();
                            let r = self.renderer.as_mut().unwrap();
                            if size.width > 0 && size.height > 0 {
                                r.resize(size.width, size.height);
                            } else {
                                r.reconfigure_surface();
                            }
                            if let Some(w) = &self.window {
                                w.request_redraw();
                            }
                            return;
                        }
                        _ => {
                            if let Some(w) = &self.window {
                                w.request_redraw();
                            }
                            return;
                        }
                    };
                    let surface_view = surface_texture
                        .texture
                        .create_view(&wgpu::TextureViewDescriptor::default());

                    {
                        let mut render_pass = encoder
                            .begin_render_pass(&wgpu::RenderPassDescriptor {
                                label: Some("egui Pass"),
                                color_attachments: &[Some(
                                    wgpu::RenderPassColorAttachment {
                                        view: &surface_view,
                                        resolve_target: None,
                                        ops: wgpu::Operations {
                                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                                r: 0.1,
                                                g: 0.1,
                                                b: 0.1,
                                                a: 1.0,
                                            }),
                                            store: wgpu::StoreOp::Store,
                                        },
                                        depth_slice: None,
                                    },
                                )],
                                depth_stencil_attachment: None,
                                ..Default::default()
                            })
                            .forget_lifetime();

                        egui_renderer.render(&mut render_pass, &tris, &screen_desc);
                    }

                    for id in &full_output.textures_delta.free {
                        egui_renderer.free_texture(id);
                    }

                    renderer.queue.submit(std::iter::once(encoder.finish()));
                    surface_texture.present();
                }

                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }

            _ => {}
        }
    }
}

/// Returns an optional path to add as a new layer (needs device access from caller).
fn build_ui(
    ctx: &egui::Context,
    layers: &mut Vec<Layer>,
    selected_layer: &mut Option<usize>,
    master_effects: &mut effects::EffectUniforms,
    master_paused: &mut bool,
    library_folder: &mut Option<PathBuf>,
    library_files: &mut Vec<PathBuf>,
    video_egui_texture_id: Option<egui::TextureId>,
    output_width: u32,
    output_height: u32,
) -> Option<String> {
    let mut add_layer_path: Option<String> = None;
    let mut remove_layer: Option<usize> = None;
    let mut move_layer: Option<(usize, usize)> = None;
    let mut change_folder = false;

    // LEFT panel: Layers + Master + Selected Layer controls
    egui::Panel::left("left_panel")
        .min_size(260.0)
        .default_size(300.0)
        .show(ctx, |ui| {
            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    // === LAYERS ===
                    ui.heading("Layers");
                    ui.separator();

                    let layer_count = layers.len();
                    for i in 0..layer_count {
                        let is_selected = *selected_layer == Some(i);
                        let row_response = ui.horizontal(|ui| {
                            // Grip handle for reorder
                            let grip = ui.label(
                                egui::RichText::new("⠿").weak(),
                            );
                            // Drag up/down via grip
                            if grip.dragged() {
                                let delta = grip.drag_delta().y;
                                if delta < -16.0 && i > 0 {
                                    move_layer = Some((i, i - 1));
                                } else if delta > 16.0 && i < layer_count - 1 {
                                    move_layer = Some((i, i + 1));
                                }
                            }

                            // Visibility toggle (eye icon)
                            let eye = if layers[i].visible { "👁" } else { "ꞏ" };
                            if ui.small_button(eye).clicked() {
                                layers[i].visible = !layers[i].visible;
                            }

                            // Layer number + pause indicator
                            let prefix = format!(
                                "{}{}.",
                                if layers[i].paused { "⏸" } else { "" },
                                i + 1
                            );
                            ui.label(egui::RichText::new(prefix).weak());

                            // Filename (selectable)
                            let response = ui.selectable_label(
                                is_selected,
                                &layers[i].filename,
                            );
                            if response.clicked() {
                                *selected_layer = Some(i);
                            }

                            // Remove button
                            if ui.small_button("×").clicked() {
                                remove_layer = Some(i);
                            }
                        });
                        // Clicking anywhere on the row also selects
                        if row_response.response.clicked() {
                            *selected_layer = Some(i);
                        }
                    }

                    if layers.is_empty() {
                        ui.weak("No active layers");
                        ui.weak("Drop files or add from library →");
                    }

                    ui.add_space(12.0);
                    ui.separator();

                    // === MASTER ===
                    ui.heading("Master");
                    ui.separator();

                    // Transport
                    ui.horizontal(|ui| {
                        if ui
                            .button(if *master_paused {
                                "▶ Play All"
                            } else {
                                "⏸ Pause All"
                            })
                            .clicked()
                        {
                            *master_paused = !*master_paused;
                        }
                        if ui.button("Reset FX").clicked() {
                            master_effects.reset();
                        }
                    });

                    ui.add_space(4.0);

                    // Master effects
                    effects_sliders(ui, master_effects, "master");

                    ui.add_space(12.0);
                    ui.separator();

                    // === SELECTED LAYER ===
                    ui.heading("Layer");
                    ui.separator();

                    if let Some(idx) = *selected_layer {
                        if let Some(layer) = layers.get_mut(idx) {
                            ui.label(
                                egui::RichText::new(&layer.filename).strong(),
                            );
                            ui.add_space(4.0);

                            // Transport
                            ui.horizontal(|ui| {
                                if ui
                                    .button(if layer.paused {
                                        "▶ Play"
                                    } else {
                                        "⏸ Pause"
                                    })
                                    .clicked()
                                {
                                    layer.paused = !layer.paused;
                                }
                                if ui.button("Reset FX").clicked() {
                                    layer.effects.reset();
                                }
                            });

                            ui.label("Speed");
                            ui.add(
                                egui::Slider::new(&mut layer.speed, 0.25..=4.0)
                                    .logarithmic(true)
                                    .custom_formatter(|v, _| format!("{:.2}×", v)),
                            );

                            ui.label("FPS");
                            ui.add(
                                egui::Slider::new(&mut layer.fps, 1.0..=60.0)
                                    .step_by(1.0)
                                    .custom_formatter(|v, _| format!("{:.0}", v)),
                            );

                            ui.add_space(4.0);

                            // Blend
                            ui.label("Opacity");
                            ui.add(
                                egui::Slider::new(&mut layer.opacity, 0.0..=1.0),
                            );

                            ui.label("Blend Mode");
                            egui::ComboBox::from_id_salt("blend_mode")
                                .selected_text(layer.blend_mode.label())
                                .show_ui(ui, |ui| {
                                    for mode in BlendMode::ALL {
                                        ui.selectable_value(
                                            &mut layer.blend_mode,
                                            *mode,
                                            mode.label(),
                                        );
                                    }
                                });

                            ui.add_space(4.0);
                            ui.separator();

                            // Per-layer effects
                            effects_sliders(ui, &mut layer.effects, "layer");
                        } else {
                            ui.weak("Invalid selection");
                        }
                    } else {
                        ui.weak("Select a layer to edit");
                    }
                });
        });

    // RIGHT panel: Library
    egui::Panel::right("right_panel")
        .min_size(160.0)
        .default_size(180.0)
        .show(ctx, |ui| {
            ui.heading("Library");
            ui.separator();

            // Folder path + change button
            ui.horizontal(|ui| {
                let folder_label = library_folder
                    .as_ref()
                    .and_then(|f| f.file_name())
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_else(|| "No folder".into());
                ui.label(folder_label);
                if ui.small_button("…").clicked() {
                    change_folder = true;
                }
            });

            ui.add_space(4.0);
            ui.separator();

            // File list
            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    for file in library_files.iter() {
                        let name = file
                            .file_name()
                            .map(|n| n.to_string_lossy().to_string())
                            .unwrap_or_default();

                        let response = ui.selectable_label(false, &name);
                        if response.double_clicked() {
                            if let Some(path_str) = file.to_str() {
                                add_layer_path = Some(path_str.to_string());
                            }
                        }
                        response.on_hover_text("Double-click to add as layer");
                    }

                    if library_files.is_empty() {
                        ui.weak("No video files");
                        ui.weak("Drop a folder or click …");
                    }
                });
        });

    // Central panel: video output
    egui::CentralPanel::default().show(ctx, |ui| {
        if let Some(tex_id) = video_egui_texture_id {
            let available = ui.available_size();
            let aspect = output_width as f32 / output_height as f32;
            let (w, h) = fit_to_area(available.x, available.y, aspect);
            ui.centered_and_justified(|ui| {
                ui.image(egui::load::SizedTexture::new(tex_id, egui::vec2(w, h)));
            });
        }
    });

    // --- Apply deferred actions ---

    // Change folder (opens native dialog)
    if change_folder {
        if let Some(folder) = rfd::FileDialog::new().pick_folder() {
            *library_files = scan_folder(&folder);
            *library_folder = Some(folder);
        }
    }

    // Move layer
    if let Some((from, to)) = move_layer {
        layers.swap(from, to);
        if *selected_layer == Some(from) {
            *selected_layer = Some(to);
        } else if *selected_layer == Some(to) {
            *selected_layer = Some(from);
        }
    }

    // Remove layer
    if let Some(idx) = remove_layer {
        layers.remove(idx);
        if layers.is_empty() {
            *selected_layer = None;
        } else if let Some(sel) = *selected_layer {
            if sel >= layers.len() {
                *selected_layer = Some(layers.len() - 1);
            }
        }
    }

    add_layer_path
}

/// Shared effects slider UI — used for both master and per-layer effects.
fn effects_sliders(ui: &mut egui::Ui, effects: &mut effects::EffectUniforms, id_prefix: &str) {
    // --- Digital effects ---
    ui.label("Pixelate");
    ui.add(
        egui::Slider::new(&mut effects.pixelate_size, 1.0..=32.0)
            .step_by(1.0)
            .custom_formatter(|v, _| format!("{:.0}", v)),
    );

    ui.label("RGB Split");
    ui.add(
        egui::Slider::new(&mut effects.rgb_split, 0.0..=30.0)
            .step_by(1.0)
            .custom_formatter(|v, _| format!("{:.0}", v)),
    );

    ui.label("Hue");
    ui.add(
        egui::Slider::new(&mut effects.hue_shift, -180.0..=180.0)
            .step_by(1.0)
            .suffix("°"),
    );

    ui.label("Saturation");
    ui.add(egui::Slider::new(&mut effects.saturation, -1.0..=1.0));

    ui.label("Brightness");
    ui.add(egui::Slider::new(&mut effects.brightness, -1.0..=1.0));

    ui.label("Contrast");
    ui.add(egui::Slider::new(&mut effects.contrast, -1.0..=1.0));

    ui.label("Posterize");
    ui.add(
        egui::Slider::new(&mut effects.posterize, 0.0..=16.0)
            .step_by(1.0)
            .custom_formatter(|v, _| {
                if v < 2.0 {
                    "Off".to_string()
                } else {
                    format!("{:.0}", v)
                }
            }),
    );

    let mut invert_on = effects.invert > 0.5;
    if ui
        .checkbox(&mut invert_on, format!("Invert##{id_prefix}"))
        .changed()
    {
        effects.invert = if invert_on { 1.0 } else { 0.0 };
    }

    ui.add_space(6.0);
    ui.separator();

    // --- Analog effects ---
    ui.label(egui::RichText::new("Analog").strong());

    ui.label("Grain");
    ui.add(egui::Slider::new(&mut effects.grain_intensity, 0.0..=0.3));

    if effects.grain_intensity > 0.0 {
        ui.horizontal(|ui| {
            ui.label("Size");
            ui.add(
                egui::Slider::new(&mut effects.grain_size, 1.0..=4.0)
                    .step_by(1.0)
                    .custom_formatter(|v, _| format!("{:.0}", v)),
            );
        });
        ui.horizontal(|ui| {
            ui.label("Algo");
            egui::ComboBox::from_id_salt(format!("grain_algo_{id_prefix}"))
                .width(90.0)
                .selected_text(match effects.grain_algo as i32 {
                    1 => "Perlin",
                    2 => "Salt&Pepper",
                    3 => "Blue",
                    _ => "Gaussian",
                })
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut effects.grain_algo, 0.0, "Gaussian");
                    ui.selectable_value(&mut effects.grain_algo, 1.0, "Perlin");
                    ui.selectable_value(&mut effects.grain_algo, 2.0, "Salt&Pepper");
                    ui.selectable_value(&mut effects.grain_algo, 3.0, "Blue");
                });
        });
        let mut color = effects.color_grain > 0.5;
        if ui.checkbox(&mut color, format!("Color grain##{id_prefix}")).changed() {
            effects.color_grain = if color { 1.0 } else { 0.0 };
        }
    }

    ui.label("Vignette");
    ui.add(egui::Slider::new(&mut effects.vignette, 0.0..=1.5));

    ui.label("Color Drift");
    ui.add(egui::Slider::new(&mut effects.color_drift, 0.0..=0.02));

    ui.label("Breathe Scale");
    ui.add(egui::Slider::new(&mut effects.breathe_scale, 0.0..=0.05));

    ui.label("Breathe Rotate");
    ui.add(
        egui::Slider::new(&mut effects.breathe_rotation, 0.0..=2.0)
            .suffix("°"),
    );

    ui.label("Breathe Drift");
    ui.add(egui::Slider::new(&mut effects.breathe_position, 0.0..=0.02));
}

/// Fit a rectangle with given aspect ratio into available width/height.
fn fit_to_area(max_w: f32, max_h: f32, aspect: f32) -> (f32, f32) {
    let w = max_w;
    let h = w / aspect;
    if h <= max_h {
        (w, h)
    } else {
        let h = max_h;
        let w = h * aspect;
        (w, h)
    }
}

fn configure_fonts(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();

    fonts.font_data.insert(
        "IBMPlexSans".to_owned(),
        Arc::new(egui::FontData::from_static(include_bytes!(
            "/Users/luis/Library/Fonts/IBMPlexSans-Regular.otf"
        ))),
    );

    fonts.font_data.insert(
        "IBMPlexMono".to_owned(),
        Arc::new(egui::FontData::from_static(include_bytes!(
            "/Users/luis/Library/Fonts/IBMPlexMono-Regular.otf"
        ))),
    );

    fonts
        .families
        .entry(egui::FontFamily::Proportional)
        .or_default()
        .insert(0, "IBMPlexSans".to_owned());

    fonts
        .families
        .entry(egui::FontFamily::Monospace)
        .or_default()
        .insert(0, "IBMPlexMono".to_owned());

    ctx.set_fonts(fonts);
}

fn main() {
    env_logger::init();

    let args: Vec<String> = std::env::args().collect();
    let arg = args.get(1).cloned();

    // Detect if arg is a folder (library) or a file (single layer)
    let (initial_video, library_folder) = match arg {
        Some(ref path) => {
            let p = PathBuf::from(path);
            if p.is_dir() {
                (None, Some(p))
            } else {
                // It's a file — also use its parent directory as the library
                let parent = p.parent().map(|p| p.to_path_buf());
                (Some(path.clone()), parent)
            }
        }
        None => {
            eprintln!("Usage: collide-o-scope [video-file-or-folder]");
            eprintln!("  Pass a video file to start with one layer,");
            eprintln!("  or a folder to browse available clips.");
            eprintln!("  You can also drag and drop files/folders onto the window.");
            (None, None)
        }
    };

    let event_loop = EventLoop::new().unwrap();
    let mut app = App::new(initial_video, library_folder);
    event_loop.run_app(&mut app).unwrap();
}

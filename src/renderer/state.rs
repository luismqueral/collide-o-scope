//! GPU rendering with wgpu.
//!
//! `Renderer` owns the GPU device and all the long-lived rendering objects, and
//! exposes a handful of methods the main loop calls each frame. A few wgpu terms
//! used throughout:
//! - **device/queue**: the GPU connection. `device` creates resources; `queue`
//!   submits work and uploads data.
//! - **surface**: the swapchain tied to the OS window — what actually gets shown.
//! - **pipeline**: a compiled (vertex + fragment shader + state) program.
//! - **bind group**: the set of textures/samplers/uniform buffers a pipeline
//!   reads, described up front by a matching *bind group layout*.
//! - **render pass**: one "draw into this target texture" operation.
//!
//! Compositing uses a ping-pong of three textures (`composite_textures`): accumulate
//! the result in `[0]`, render the next layer into `[1]`, blend `[0]+[1]` into `[2]`,
//! then copy `[2]` back to `[0]`. See `render_layers` for the full dance.
use std::sync::Arc;
use wgpu::util::DeviceExt;
use winit::window::Window;

use crate::effects::EffectUniforms;
use crate::layers::Layer;

/// Per-layer settings handed to the composite shader. `#[repr(C)]` forces a
/// predictable C-style memory layout, and the bytemuck derives let us reinterpret
/// the struct as raw bytes for upload. `_pad` rounds the size up to 16 bytes,
/// the alignment GPUs require for uniform buffers.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CompositeUniforms {
    opacity: f32,
    blend_mode: u32,
    _pad: [f32; 2],
}

pub struct Renderer {
    pub surface: wgpu::Surface<'static>,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub config: wgpu::SurfaceConfiguration,

    // Effects pipeline (per-layer: applies pixelate/rgb_split/color to a single layer)
    effects_pipeline: wgpu::RenderPipeline,
    effects_bind_group_layout: wgpu::BindGroupLayout,
    effects_uniform_layout: wgpu::BindGroupLayout,

    // Composite pipeline (blends overlay onto base)
    composite_pipeline: wgpu::RenderPipeline,
    composite_bind_group_layout: wgpu::BindGroupLayout,
    composite_uniform_layout: wgpu::BindGroupLayout,

    // Shared sampler
    sampler: wgpu::Sampler,

    // Three textures for compositing:
    // [0] = accumulated result (base)
    // [1] = current layer after effects (overlay)
    // [2] = composite output (becomes new base)
    pub composite_textures: [wgpu::Texture; 3],
    pub composite_views: [wgpu::TextureView; 3],

    // The view egui displays (always points at the final accumulated result)
    pub output_view: wgpu::TextureView,

    // Previous frame's final output, fed back into the effects shader for
    // datamosh trails. Captured from composite_textures[0] each frame start.
    feedback_texture: wgpu::Texture,
    feedback_view: wgpu::TextureView,

    pub output_width: u32,
    pub output_height: u32,

    // Persistent staging buffer for NTSC readback (avoids per-frame allocation)
    readback_buffer: Option<wgpu::Buffer>,
    readback_buffer_size: u64,

    // Half-res blit pipeline + intermediate texture for the live NTSC pass.
    // The live VHS post-process runs on half-res pixels: composite[0] is
    // downscaled into ntsc_half_texture (GPU bilinear), read back, processed by
    // ntsc-rs, written back, then upscaled to composite[0] for display.
    blit_pipeline: wgpu::RenderPipeline,
    blit_bind_group_layout: wgpu::BindGroupLayout,
    ntsc_half_texture: wgpu::Texture,
    ntsc_half_view: wgpu::TextureView,
}

impl Renderer {
    /// Build the whole GPU stack: connect to a GPU, configure the window
    /// surface, compile the three shader pipelines, and allocate the working
    /// textures. Called once at startup. Returns `Self` (the fully-built renderer).
    pub fn new(window: Arc<Window>, output_width: u32, output_height: u32) -> Self {
        let size = window.inner_size();
        // wgpu's startup handshake: instance → surface → adapter (a physical GPU)
        // → device + queue (our logical handle to it). The `request_*` calls are
        // async; `pollster::block_on` runs them to completion on this thread since
        // startup doesn't need to stay non-blocking.
        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(window).unwrap();

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .expect("No suitable GPU adapter found");

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("Device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            ..Default::default()
        }))
        .expect("Failed to create device");

        // Pick an sRGB surface format if the GPU offers one (correct gamma),
        // else fall back to whatever's first. `iter().find(...)` scans the list;
        // `.copied()` turns the `&format` reference into an owned value.
        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        // How the swapchain behaves. `Fifo` = vsync (cap to refresh rate, no
        // tearing); `RENDER_ATTACHMENT` = we draw directly into it.
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // --- Effects pipeline (single texture + uniforms → render target) ---
        // A bind group *layout* declares the shape of resources a shader expects
        // (here: binding 0 = the layer texture, 1 = the sampler, 2 = the previous
        // frame for datamosh feedback). The actual textures are bound per-frame
        // via matching "bind groups". `visibility: FRAGMENT` = only the fragment
        // shader reads these.
        let effects_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Effects Texture BGL"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    // Previous output frame (datamosh feedback)
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                ],
            });

        let effects_uniform_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Effects Uniform BGL"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        // Shaders are compiled from WGSL source. `include_str!` embeds the file
        // contents into the binary at *compile* time (so there's no runtime file
        // read). One vertex shader is shared by all three pipelines: it emits a
        // single triangle big enough to cover the screen, so the fragment shader
        // runs once per output pixel.
        let vertex_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Vertex Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/fullscreen.wgsl").into()),
        });

        let effects_fragment = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Effects Fragment"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/effects.wgsl").into()),
        });

        let effects_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Effects Pipeline Layout"),
                bind_group_layouts: &[
                    Some(&effects_bind_group_layout),
                    Some(&effects_uniform_layout),
                ],
                immediate_size: 0,
            });

        // A render pipeline ties together the shaders + fixed-function state into
        // one compiled GPU program. `buffers: &[]` = no vertex buffer (the shader
        // generates the triangle from `vertex_index`); `blend: REPLACE` (below)
        // = overwrite the target, since blending is done in our own shader maths.
        let effects_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Effects Pipeline"),
            layout: Some(&effects_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &vertex_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &effects_fragment,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // --- Composite pipeline (two textures + uniforms → render target) ---
        let composite_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Composite BGL"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let composite_uniform_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Composite Uniform BGL"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let composite_fragment = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Composite Fragment"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/composite.wgsl").into()),
        });

        let composite_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Composite Pipeline Layout"),
                bind_group_layouts: &[
                    Some(&composite_bind_group_layout),
                    Some(&composite_uniform_layout),
                ],
                immediate_size: 0,
            });

        let composite_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Composite Pipeline"),
            layout: Some(&composite_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &vertex_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &composite_fragment,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // --- Blit pipeline (passthrough; used as GPU bilinear down/upscale for NTSC) ---
        let blit_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Blit BGL"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let blit_fragment = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Blit Fragment"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/blit.wgsl").into()),
        });

        let blit_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Blit Pipeline Layout"),
            bind_group_layouts: &[Some(&blit_bind_group_layout)],
            immediate_size: 0,
        });

        let blit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Blit Pipeline"),
            layout: Some(&blit_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &vertex_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &blit_fragment,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // --- Three composite textures ---
        // All four usage flags because these textures play every role: drawn
        // into (RENDER_ATTACHMENT), sampled by shaders (TEXTURE_BINDING), and
        // copied both directions (COPY_SRC/COPY_DST) for the ping-pong + NTSC
        // readback. `std::array::from_fn(|i| ...)` builds the `[T; 3]` array by
        // calling the closure once per index — handy when each element needs
        // its own GPU allocation (you can't `Copy` a texture).
        let tex_usage = wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST;

        let composite_textures: [wgpu::Texture; 3] = std::array::from_fn(|i| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(&format!("Composite {i}")),
                size: wgpu::Extent3d {
                    width: output_width,
                    height: output_height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: tex_usage,
                view_formats: &[],
            })
        });

        let composite_views: [wgpu::TextureView; 3] = std::array::from_fn(|i| {
            composite_textures[i].create_view(&wgpu::TextureViewDescriptor::default())
        });

        let output_view =
            composite_textures[0].create_view(&wgpu::TextureViewDescriptor::default());

        // Feedback texture: holds the previous frame's final output for datamosh.
        let feedback_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Feedback (prev frame)"),
            size: wgpu::Extent3d {
                width: output_width,
                height: output_height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: tex_usage,
            view_formats: &[],
        });
        let feedback_view = feedback_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Half-res intermediate for the live NTSC pass.
        let half_w = (output_width / 2).max(1);
        let half_h = (output_height / 2).max(1);
        let ntsc_half_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("NTSC Half"),
            size: wgpu::Extent3d {
                width: half_w,
                height: half_h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: tex_usage,
            view_formats: &[],
        });
        let ntsc_half_view = ntsc_half_texture.create_view(&wgpu::TextureViewDescriptor::default());

        Self {
            surface,
            device,
            queue,
            config,
            effects_pipeline,
            effects_bind_group_layout,
            effects_uniform_layout,
            composite_pipeline,
            composite_bind_group_layout,
            composite_uniform_layout,
            sampler,
            composite_textures,
            composite_views,
            output_view,
            feedback_texture,
            feedback_view,
            output_width,
            output_height,
            readback_buffer: None,
            readback_buffer_size: 0,
            blit_pipeline,
            blit_bind_group_layout,
            ntsc_half_texture,
            ntsc_half_view,
        }
    }

    pub fn resize(&mut self, new_width: u32, new_height: u32) {
        if new_width > 0 && new_height > 0 {
            self.config.width = new_width;
            self.config.height = new_height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    /// Resize the offscreen composite canvas (live output + export source).
    /// Recreates the 3 composite textures + feedback texture at w×h. Pipelines,
    /// layouts, and the sampler are size-independent; per-frame bind groups are
    /// rebuilt in render_layers/render_master_effects, so nothing else changes.
    /// Caller must rebind the egui preview texture afterwards (output_view moves).
    pub fn set_output_size(&mut self, w: u32, h: u32) {
        if w == 0 || h == 0 || (w == self.output_width && h == self.output_height) {
            return;
        }
        let tex_usage = wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST;

        // Build into locals first (the from_fn closures borrow &self.device,
        // so we can't assign into self fields inside them).
        let composite_textures: [wgpu::Texture; 3] = std::array::from_fn(|i| {
            self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(&format!("Composite {i}")),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: tex_usage,
                view_formats: &[],
            })
        });
        let composite_views: [wgpu::TextureView; 3] = std::array::from_fn(|i| {
            composite_textures[i].create_view(&wgpu::TextureViewDescriptor::default())
        });
        let output_view =
            composite_textures[0].create_view(&wgpu::TextureViewDescriptor::default());

        let feedback_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Feedback (prev frame)"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: tex_usage,
            view_formats: &[],
        });
        let feedback_view = feedback_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Half-res NTSC intermediate tracks the new size.
        let half_w = (w / 2).max(1);
        let half_h = (h / 2).max(1);
        let ntsc_half_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("NTSC Half"),
            size: wgpu::Extent3d {
                width: half_w,
                height: half_h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: tex_usage,
            view_formats: &[],
        });
        let ntsc_half_view = ntsc_half_texture.create_view(&wgpu::TextureViewDescriptor::default());

        self.composite_textures = composite_textures;
        self.composite_views = composite_views;
        self.output_view = output_view;
        self.feedback_texture = feedback_texture;
        self.feedback_view = feedback_view;
        self.ntsc_half_texture = ntsc_half_texture;
        self.ntsc_half_view = ntsc_half_view;
        self.output_width = w;
        self.output_height = h;
        // readback_buffer auto-grows in readback_composite(); no reset needed.
    }

    /// Force reconfigure the surface (e.g. after Lost/Outdated during fullscreen transition).
    pub fn reconfigure_surface(&self) {
        self.surface.configure(&self.device, &self.config);
    }

    /// Render all layers composited together. Final result ends up in composite_views[0].
    /// Each layer gets its own effects applied.
    pub fn render_layers(&self, encoder: &mut wgpu::CommandEncoder, layers: &[Layer]) {
        // Capture last frame's final output (still in [0]) for datamosh feedback,
        // before this frame's passes overwrite it.
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.composite_textures[0],
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: &self.feedback_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: self.output_width,
                height: self.output_height,
                depth_or_array_layers: 1,
            },
        );

        if layers.is_empty() {
            // Clear to black
            encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Clear"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.composite_views[0],
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            return;
        }

        // Render in reverse order: last layer in the vec is the bottom,
        // first layer (index 0, "Layer 1" in UI) ends up on top. Audio-only
        // layers have only a 1×1 placeholder texture, so they're excluded.
        let visible_layers: Vec<&Layer> = layers
            .iter()
            .filter(|l| l.visible && !l.audio_only)
            .rev()
            .collect();

        // Each layer is handled in two steps: (1) run its effect shader into the
        // scratch texture [1], then (2) blend [1] over the accumulator [0]. The
        // very first (bottom) layer skips the blend and is copied straight into
        // [0] as the starting canvas; every later layer composites on top.
        for (i, layer) in visible_layers.iter().enumerate() {
            let uniforms = layer.effects;

            // Each pass needs its own buffer because queue.write_buffer writes
            // all execute before the encoder's render passes run on the GPU.
            let fx_buffer = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Layer FX Uniforms"),
                    contents: bytemuck::cast_slice(&[uniforms]),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

            let tex_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.effects_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&layer.texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&self.feedback_view),
                    },
                ],
            });

            let uniform_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.effects_uniform_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: fx_buffer.as_entire_binding(),
                }],
            });

            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Layer FX"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.composite_views[1],
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    ..Default::default()
                });
                pass.set_pipeline(&self.effects_pipeline);
                pass.set_bind_group(0, &tex_bg, &[]);
                pass.set_bind_group(1, &uniform_bg, &[]);
                pass.draw(0..3, 0..1);
            }

            // Step 2: Composite layer onto accumulated result
            if i == 0 {
                // First layer: copy effects result directly to accumulator [0]
                encoder.copy_texture_to_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &self.composite_textures[1],
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::TexelCopyTextureInfo {
                        texture: &self.composite_textures[0],
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::Extent3d {
                        width: self.output_width,
                        height: self.output_height,
                        depth_or_array_layers: 1,
                    },
                );
            } else {
                // Subsequent layers: composite base[0] + overlay[1] → temp[2]
                let comp_buffer =
                    self.device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("Composite Uniforms"),
                            contents: bytemuck::cast_slice(&[CompositeUniforms {
                                opacity: layer.opacity,
                                blend_mode: layer.blend_mode.as_u32(),
                                _pad: [0.0; 2],
                            }]),
                            usage: wgpu::BufferUsages::UNIFORM,
                        });

                let composite_tex_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Composite Textures BG"),
                    layout: &self.composite_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&self.composite_views[0]),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&self.composite_views[1]),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(&self.sampler),
                        },
                    ],
                });

                let composite_uniform_bg =
                    self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("Composite Uniform BG"),
                        layout: &self.composite_uniform_layout,
                        entries: &[wgpu::BindGroupEntry {
                            binding: 0,
                            resource: comp_buffer.as_entire_binding(),
                        }],
                    });

                // Render composite to [2]
                {
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Composite Pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &self.composite_views[2],
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                store: wgpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        })],
                        depth_stencil_attachment: None,
                        ..Default::default()
                    });
                    pass.set_pipeline(&self.composite_pipeline);
                    pass.set_bind_group(0, &composite_tex_bg, &[]);
                    pass.set_bind_group(1, &composite_uniform_bg, &[]);
                    pass.draw(0..3, 0..1);
                }

                // Copy [2] → [0] so it becomes the new accumulated base
                encoder.copy_texture_to_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &self.composite_textures[2],
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::TexelCopyTextureInfo {
                        texture: &self.composite_textures[0],
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::Extent3d {
                        width: self.output_width,
                        height: self.output_height,
                        depth_or_array_layers: 1,
                    },
                );
            }
        }
    }

    /// Apply master effects to the final composite (already in [0]).
    /// Reads [0], applies effects → [2], copies back to [0].
    pub fn render_master_effects(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        master_uniforms: &EffectUniforms,
    ) {
        let fx_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Master FX Uniforms"),
                contents: bytemuck::cast_slice(&[*master_uniforms]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let tex_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Master FX Input"),
            layout: &self.effects_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.composite_views[0]),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&self.feedback_view),
                },
            ],
        });

        let uniform_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Master FX Uniforms BG"),
            layout: &self.effects_uniform_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: fx_buffer.as_entire_binding(),
            }],
        });

        // Render effects from [0] → [2]
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Master FX Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.composite_views[2],
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            pass.set_pipeline(&self.effects_pipeline);
            pass.set_bind_group(0, &tex_bg, &[]);
            pass.set_bind_group(1, &uniform_bg, &[]);
            pass.draw(0..3, 0..1);
        }

        // Copy [2] → [0]
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.composite_textures[2],
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: &self.composite_textures[0],
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: self.output_width,
                height: self.output_height,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Read a texture back to CPU as tightly-packed RGBA bytes (w*4 per row).
    /// Submits a copy to the GPU and blocks until it completes. Uses a persistent
    /// staging buffer (auto-grows) to avoid per-frame allocation.
    fn readback_texture(&mut self, tex: &wgpu::Texture, w: u32, h: u32) -> Vec<u8> {
        // wgpu requires each row in a texture→buffer copy to start on a 256-byte
        // boundary. `(n + 255) & !255` is the standard "round up to a multiple of
        // 256" trick: add 255 then mask off the low 8 bits. The padding is
        // stripped back out below before the pixels are returned.
        let bytes_per_row = (w * 4 + 255) & !255;
        let buffer_size = (bytes_per_row * h) as u64;

        // Reuse or create staging buffer
        if self.readback_buffer.is_none() || self.readback_buffer_size < buffer_size {
            self.readback_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("NTSC Readback"),
                size: buffer_size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }));
            self.readback_buffer_size = buffer_size;
        }
        let staging = self.readback_buffer.as_ref().unwrap();

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("NTSC Readback Encoder"),
            });

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        // GPU memory isn't directly readable by the CPU — you have to "map" it
        // first, and mapping is asynchronous. The standard blocking pattern:
        // kick off `map_async` (which signals a channel when ready), then
        // `poll(Wait)` to drive the GPU to completion, then `rx.recv()` blocks
        // until the callback has fired. After that the bytes are safe to read.
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        // Remove row padding if any
        let row_bytes = (w * 4) as usize;
        let padded_row = bytes_per_row as usize;
        let mut pixels = Vec::with_capacity(row_bytes * h as usize);
        for row in 0..h as usize {
            let start = row * padded_row;
            pixels.extend_from_slice(&data[start..start + row_bytes]);
        }
        drop(data);
        staging.unmap();

        pixels
    }

    /// Read composite_textures[0] back to CPU as RGBA bytes (full output res).
    pub fn readback_composite(&mut self) -> Vec<u8> {
        let (w, h) = (self.output_width, self.output_height);
        // Clone the Arc-backed texture handle so the &self borrow ends before &mut self.
        let tex = self.composite_textures[0].clone();
        self.readback_texture(&tex, w, h)
    }

    /// Read the half-res NTSC texture back to CPU as RGBA bytes.
    pub fn readback_half(&mut self) -> Vec<u8> {
        let w = (self.output_width / 2).max(1);
        let h = (self.output_height / 2).max(1);
        let tex = self.ntsc_half_texture.clone();
        self.readback_texture(&tex, w, h)
    }

    /// Write RGBA pixels back to composite_textures[0].
    pub fn write_composite(&self, pixels: &[u8]) {
        let w = self.output_width;
        let h = self.output_height;
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.composite_textures[0],
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(w * 4),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Write RGBA pixels back into the half-res NTSC texture.
    pub fn write_half(&self, pixels: &[u8]) {
        let w = (self.output_width / 2).max(1);
        let h = (self.output_height / 2).max(1);
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.ntsc_half_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(w * 4),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
    }

    /// One-shot passthrough blit: render `src_view` into `dst_view`. The shared
    /// sampler is Linear, so this acts as a GPU bilinear down/upscale when the
    /// source and target differ in size.
    fn blit(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        src_view: &wgpu::TextureView,
        dst_view: &wgpu::TextureView,
    ) {
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Blit BG"),
            layout: &self.blit_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(src_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Blit Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: dst_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            ..Default::default()
        });
        pass.set_pipeline(&self.blit_pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.draw(0..3, 0..1);
    }

    /// Downscale composite_textures[0] into the half-res NTSC texture (GPU bilinear).
    pub fn downscale_to_half(&self, encoder: &mut wgpu::CommandEncoder) {
        let src = self.composite_textures[0].create_view(&wgpu::TextureViewDescriptor::default());
        self.blit(encoder, &src, &self.ntsc_half_view);
    }

    /// Upscale the half-res NTSC texture back into composite_textures[0] (GPU bilinear).
    pub fn upscale_half_to_composite(&self, encoder: &mut wgpu::CommandEncoder) {
        self.blit(encoder, &self.ntsc_half_view, &self.output_view);
    }
}

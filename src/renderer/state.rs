use std::sync::Arc;
use wgpu::util::DeviceExt;
use winit::window::Window;

use crate::effects::EffectUniforms;
use crate::layers::Layer;

/// Uniforms for the composite shader.
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

    pub output_width: u32,
    pub output_height: u32,
}

impl Renderer {
    pub fn new(window: Arc<Window>, output_width: u32, output_height: u32) -> Self {
        let size = window.inner_size();
        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(window).unwrap();

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .expect("No suitable GPU adapter found");

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                ..Default::default()
            },
        ))
        .expect("Failed to create device");

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

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

        // --- Three composite textures ---
        let tex_usage =
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST;

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
            output_width,
            output_height,
        }
    }

    pub fn resize(&mut self, new_width: u32, new_height: u32) {
        if new_width > 0 && new_height > 0 {
            self.config.width = new_width;
            self.config.height = new_height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    /// Force reconfigure the surface (e.g. after Lost/Outdated during fullscreen transition).
    pub fn reconfigure_surface(&self) {
        self.surface.configure(&self.device, &self.config);
    }

    /// Render all layers composited together. Final result ends up in composite_views[0].
    /// Each layer gets its own effects applied.
    pub fn render_layers(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        layers: &[Layer],
    ) {
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
        // first layer (index 0, "Layer 1" in UI) ends up on top.
        let visible_layers: Vec<&Layer> = layers.iter().filter(|l| l.visible).rev().collect();

        for (i, layer) in visible_layers.iter().enumerate() {
            let uniforms = layer.effects;

            // Each pass needs its own buffer because queue.write_buffer writes
            // all execute before the encoder's render passes run on the GPU.
            let fx_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
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
                    self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Composite Uniforms"),
                        contents: bytemuck::cast_slice(&[CompositeUniforms {
                            opacity: layer.opacity,
                            blend_mode: layer.blend_mode.as_u32(),
                            _pad: [0.0; 2],
                        }]),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });

                let composite_tex_bg =
                    self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("Composite Textures BG"),
                        layout: &self.composite_bind_group_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(
                                    &self.composite_views[0],
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(
                                    &self.composite_views[1],
                                ),
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
        let fx_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
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
}

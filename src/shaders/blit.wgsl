// Passthrough blit fragment shader — samples a source texture into the target.
// Paired with fullscreen.wgsl's vs_main (UV at @location(0)). The bound sampler
// is Linear/Linear, so this doubles as a GPU bilinear down/upscale.

@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;

@fragment
fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
    return textureSample(src_tex, samp, uv);
}

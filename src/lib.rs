//! Two-pass GPU GLSL ordered-dither (Bayer 8x8) interactive background for `egui` / `eframe`
//! using the `glow` backend.
//!
//! Renders either drifting 4-octave value-noise clouds (`DitherVariant::Field`) or a soft
//! breathing organic form (`DitherVariant::Bloom`) with a continuous cursor trail brush and
//! expanding click ripples.
//!
//! Work per frame on the CPU is a small set of uniforms. Pass 1 evaluates noise, cursor trail,
//! and click ripples once per dither cell into a tiny offscreen RGBA8 texture; Pass 2 composites
//! the Bayer 8x8 threshold, square pixel mask, and soft vignette at full resolution.

use eframe::{egui, egui_glow, glow};
use std::cell::RefCell;
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

pub const TRAIL: usize = 16;
pub const RIPPLES: usize = 6;
pub const TRAIL_SECONDS: f64 = 0.9;
pub const RIPPLE_SECONDS: f64 = 2.4;

/// Set when the GPU path cannot run (GLSL 1.20 / ES 2 contexts or a compile failure);
/// callers can check [`available`] to fall back to a CPU background.
static UNAVAILABLE: AtomicBool = AtomicBool::new(false);

/// Returns `true` if the OpenGL / glow shader path compiled and is available.
pub fn available() -> bool {
    !UNAVAILABLE.load(Ordering::Relaxed)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DitherVariant {
    Field,
    Bloom,
}

/// Everything the shader needs for one frame, in points relative to the background rect.
#[derive(Clone, Copy, Debug)]
pub struct DitherFrame {
    pub variant: DitherVariant,
    pub time: f32,
    pub ink: egui::Rgba,
    pub accent: egui::Rgba,
    pub strength: f32,
    pub density: f32,
    pub pixel: f32,
    pub light: bool,
    /// `(x, y, age 0..1)` per trail sample; `age >= 1.0` means unused.
    pub trail: [[f32; 3]; TRAIL],
    /// `(x, y, seconds since click)`; negative seconds means unused.
    pub ripples: [[f32; 3]; RIPPLES],
}

impl Default for DitherFrame {
    fn default() -> Self {
        Self {
            variant: DitherVariant::Field,
            time: 0.0,
            ink: egui::Rgba::from_rgb(0.62, 0.65, 0.72),
            accent: egui::Rgba::from_rgb(0.42, 0.69, 1.0),
            strength: 0.9,
            density: 0.55,
            pixel: 4.0,
            light: false,
            trail: [[0.0, 0.0, 1.0]; TRAIL],
            ripples: [[0.0, 0.0, -1.0]; RIPPLES],
        }
    }
}

/// Pointer and click ripple history kept on the UI thread.
#[derive(Default)]
pub struct DitherInput {
    trail: VecDeque<(egui::Pos2, f64)>,
    ripples: VecDeque<(egui::Pos2, f64)>,
}

impl DitherInput {
    pub fn record(&mut self, now: f64, pointer: Option<egui::Pos2>, click: bool) {
        if let Some(p) = pointer
            && self
                .trail
                .back()
                .is_none_or(|(last, _)| last.distance(p) > 3.0)
        {
            self.trail.push_back((p, now));
        }
        if click && let Some(p) = pointer {
            self.ripples.push_back((p, now));
        }
        while self.trail.len() > TRAIL
            || self
                .trail
                .front()
                .is_some_and(|(_, t)| now - t > TRAIL_SECONDS)
        {
            self.trail.pop_front();
        }
        while self.ripples.len() > RIPPLES
            || self
                .ripples
                .front()
                .is_some_and(|(_, t)| now - t > RIPPLE_SECONDS)
        {
            self.ripples.pop_front();
        }
    }

    pub fn clear(&mut self) {
        self.trail.clear();
        self.ripples.clear();
    }

    pub fn busy(&self) -> bool {
        !self.trail.is_empty() || !self.ripples.is_empty()
    }

    pub fn fill(&self, frame: &mut DitherFrame, rect: egui::Rect, now: f64) {
        frame.trail = [[0.0, 0.0, 1.0]; TRAIL];
        frame.ripples = [[0.0, 0.0, -1.0]; RIPPLES];
        for (slot, (p, t)) in frame.trail.iter_mut().zip(self.trail.iter().rev()) {
            *slot = [
                p.x - rect.left(),
                p.y - rect.top(),
                ((now - t) / TRAIL_SECONDS) as f32,
            ];
        }
        for (slot, (p, t)) in frame.ripples.iter_mut().zip(self.ripples.iter().rev()) {
            *slot = [p.x - rect.left(), p.y - rect.top(), (now - t) as f32];
        }
    }
}

/// Schedules the two-pass GLSL dither draw inside `rect` on the given `egui::Painter`.
pub fn paint(painter: &egui::Painter, rect: egui::Rect, frame: DitherFrame) {
    let callback = egui::PaintCallback {
        rect,
        callback: Arc::new(egui_glow::CallbackFn::new(move |info, glow_painter| {
            draw(info, glow_painter.gl(), &frame);
        })),
    };
    painter.add(callback);
}

struct FieldUniforms {
    size: Option<glow::UniformLocation>,
    scale: Option<glow::UniformLocation>,
    time: Option<glow::UniformLocation>,
    pixel: Option<glow::UniformLocation>,
    rows: Option<glow::UniformLocation>,
    density: Option<glow::UniformLocation>,
    variant: Option<glow::UniformLocation>,
    trail: Option<glow::UniformLocation>,
    ripples: Option<glow::UniformLocation>,
}

struct ComposeUniforms {
    cells: Option<glow::UniformLocation>,
    origin: Option<glow::UniformLocation>,
    size: Option<glow::UniformLocation>,
    pixel: Option<glow::UniformLocation>,
    cell_count: Option<glow::UniformLocation>,
    strength: Option<glow::UniformLocation>,
    light: Option<glow::UniformLocation>,
    ink: Option<glow::UniformLocation>,
    accent: Option<glow::UniformLocation>,
}

struct Gpu {
    field: glow::Program,
    field_u: FieldUniforms,
    compose: glow::Program,
    compose_u: ComposeUniforms,
    vao: glow::VertexArray,
    texture: glow::Texture,
    framebuffer: glow::Framebuffer,
    cells: (i32, i32),
}

thread_local! {
    static GPU: RefCell<Option<Gpu>> = const { RefCell::new(None) };
}

fn draw(info: egui::PaintCallbackInfo, gl: &Arc<glow::Context>, frame: &DitherFrame) {
    use glow::HasContext as _;
    if !available() {
        return;
    }
    GPU.with(|cell| {
        let mut slot = cell.borrow_mut();
        if slot.is_none() {
            match unsafe { compile(gl) } {
                Ok(gpu) => *slot = Some(gpu),
                Err(error) => {
                    eprintln!("egui-dither-bg unavailable: {error}");
                    UNAVAILABLE.store(true, Ordering::Relaxed);
                    return;
                }
            }
        }
        let Some(gpu) = slot.as_mut() else { return };
        let viewport = info.viewport_in_pixels();
        let clip = info.clip_rect_in_pixels();
        let ppp = info.pixels_per_point;
        let pixel = (frame.pixel * ppp).round().max(2.0);
        let cells = (
            (viewport.width_px as f32 / pixel).ceil().max(1.0) as i32,
            (viewport.height_px as f32 / pixel).ceil().max(1.0) as i32,
        );
        unsafe {
            if gpu.cells != cells {
                gl.bind_texture(glow::TEXTURE_2D, Some(gpu.texture));
                gl.tex_image_2d(
                    glow::TEXTURE_2D,
                    0,
                    glow::RGBA8 as i32,
                    cells.0,
                    cells.1,
                    0,
                    glow::RGBA,
                    glow::UNSIGNED_BYTE,
                    glow::PixelUnpackData::Slice(None),
                );
                gpu.cells = cells;
            }
            let previous = gl.get_parameter_framebuffer(glow::DRAW_FRAMEBUFFER_BINDING);
            // Pass 1: field per cell into the offscreen texture.
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(gpu.framebuffer));
            gl.viewport(0, 0, cells.0, cells.1);
            gl.disable(glow::SCISSOR_TEST);
            gl.disable(glow::BLEND);
            gl.use_program(Some(gpu.field));
            let fu = &gpu.field_u;
            gl.uniform_2_f32(
                fu.size.as_ref(),
                viewport.width_px as f32,
                viewport.height_px as f32,
            );
            gl.uniform_1_f32(fu.scale.as_ref(), ppp);
            gl.uniform_1_f32(fu.time.as_ref(), frame.time);
            gl.uniform_1_f32(fu.pixel.as_ref(), pixel);
            gl.uniform_1_f32(fu.rows.as_ref(), cells.1 as f32);
            gl.uniform_1_f32(fu.density.as_ref(), frame.density);
            gl.uniform_1_f32(
                fu.variant.as_ref(),
                if frame.variant == DitherVariant::Bloom {
                    1.0
                } else {
                    0.0
                },
            );
            let mut trail = [0.0f32; TRAIL * 3];
            for (i, sample) in frame.trail.iter().enumerate() {
                trail[i * 3..i * 3 + 3].copy_from_slice(sample);
            }
            let mut ripples = [0.0f32; RIPPLES * 3];
            for (i, sample) in frame.ripples.iter().enumerate() {
                ripples[i * 3..i * 3 + 3].copy_from_slice(sample);
            }
            gl.uniform_3_f32_slice(fu.trail.as_ref(), &trail);
            gl.uniform_3_f32_slice(fu.ripples.as_ref(), &ripples);
            gl.bind_vertex_array(Some(gpu.vao));
            gl.draw_arrays(glow::TRIANGLES, 0, 3);
            // Pass 2: compose the dithered pixels into egui's framebuffer.
            gl.bind_framebuffer(glow::FRAMEBUFFER, previous);
            gl.viewport(
                viewport.left_px,
                viewport.from_bottom_px,
                viewport.width_px,
                viewport.height_px,
            );
            gl.enable(glow::SCISSOR_TEST);
            gl.scissor(
                clip.left_px,
                clip.from_bottom_px,
                clip.width_px,
                clip.height_px,
            );
            gl.enable(glow::BLEND);
            gl.use_program(Some(gpu.compose));
            let cu = &gpu.compose_u;
            gl.active_texture(glow::TEXTURE0);
            gl.bind_texture(glow::TEXTURE_2D, Some(gpu.texture));
            gl.uniform_1_i32(cu.cells.as_ref(), 0);
            gl.uniform_2_f32(
                cu.origin.as_ref(),
                viewport.left_px as f32,
                viewport.from_bottom_px as f32,
            );
            gl.uniform_2_f32(
                cu.size.as_ref(),
                viewport.width_px as f32,
                viewport.height_px as f32,
            );
            gl.uniform_1_f32(cu.pixel.as_ref(), pixel);
            gl.uniform_2_i32(cu.cell_count.as_ref(), cells.0, cells.1);
            gl.uniform_1_f32(cu.strength.as_ref(), frame.strength);
            gl.uniform_1_f32(cu.light.as_ref(), if frame.light { 1.0 } else { 0.0 });
            gl.uniform_3_f32(cu.ink.as_ref(), frame.ink.r(), frame.ink.g(), frame.ink.b());
            gl.uniform_3_f32(
                cu.accent.as_ref(),
                frame.accent.r(),
                frame.accent.g(),
                frame.accent.b(),
            );
            gl.draw_arrays(glow::TRIANGLES, 0, 3);
            gl.bind_vertex_array(None);
        }
    });
}

unsafe fn compile(gl: &glow::Context) -> Result<Gpu, String> {
    use glow::HasContext as _;
    let version = egui_glow::ShaderVersion::get(gl);
    if !version.is_new_shader_interface() {
        return Err(format!("{version:?} has no gl_VertexID"));
    }
    let header = format!(
        "{}{}",
        version.version_declaration(),
        if matches!(version, egui_glow::ShaderVersion::Es300) {
            "precision highp float;\nprecision highp int;\n"
        } else {
            ""
        }
    );
    let link = |fragment: &str| -> Result<glow::Program, String> {
        unsafe {
            let program = gl.create_program()?;
            let mut shaders = Vec::new();
            for (kind, body) in [
                (glow::VERTEX_SHADER, VERTEX),
                (glow::FRAGMENT_SHADER, fragment),
            ] {
                let shader = gl.create_shader(kind)?;
                gl.shader_source(shader, &format!("{header}{body}"));
                gl.compile_shader(shader);
                if !gl.get_shader_compile_status(shader) {
                    return Err(gl.get_shader_info_log(shader));
                }
                gl.attach_shader(program, shader);
                shaders.push(shader);
            }
            gl.link_program(program);
            if !gl.get_program_link_status(program) {
                return Err(gl.get_program_info_log(program));
            }
            for shader in shaders {
                gl.detach_shader(program, shader);
                gl.delete_shader(shader);
            }
            Ok(program)
        }
    };
    let field = link(FIELD)?;
    let compose = link(COMPOSE)?;
    unsafe {
        let vao = gl.create_vertex_array()?;
        let texture = gl.create_texture()?;
        gl.bind_texture(glow::TEXTURE_2D, Some(texture));
        for (key, value) in [
            (glow::TEXTURE_MIN_FILTER, glow::NEAREST),
            (glow::TEXTURE_MAG_FILTER, glow::NEAREST),
            (glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE),
            (glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE),
        ] {
            gl.tex_parameter_i32(glow::TEXTURE_2D, key, value as i32);
        }
        gl.tex_image_2d(
            glow::TEXTURE_2D,
            0,
            glow::RGBA8 as i32,
            1,
            1,
            0,
            glow::RGBA,
            glow::UNSIGNED_BYTE,
            glow::PixelUnpackData::Slice(None),
        );
        let previous = gl.get_parameter_framebuffer(glow::DRAW_FRAMEBUFFER_BINDING);
        let framebuffer = gl.create_framebuffer()?;
        gl.bind_framebuffer(glow::FRAMEBUFFER, Some(framebuffer));
        gl.framebuffer_texture_2d(
            glow::FRAMEBUFFER,
            glow::COLOR_ATTACHMENT0,
            glow::TEXTURE_2D,
            Some(texture),
            0,
        );
        let complete = gl.check_framebuffer_status(glow::FRAMEBUFFER) == glow::FRAMEBUFFER_COMPLETE;
        gl.bind_framebuffer(glow::FRAMEBUFFER, previous);
        if !complete {
            return Err("offscreen framebuffer incomplete".into());
        }
        let field_u = FieldUniforms {
            size: gl.get_uniform_location(field, "uSize"),
            scale: gl.get_uniform_location(field, "uScale"),
            time: gl.get_uniform_location(field, "uTime"),
            pixel: gl.get_uniform_location(field, "uPixel"),
            rows: gl.get_uniform_location(field, "uRows"),
            density: gl.get_uniform_location(field, "uDensity"),
            variant: gl.get_uniform_location(field, "uVariant"),
            trail: gl.get_uniform_location(field, "uTrail[0]"),
            ripples: gl.get_uniform_location(field, "uRipples[0]"),
        };
        let compose_u = ComposeUniforms {
            cells: gl.get_uniform_location(compose, "uCells"),
            origin: gl.get_uniform_location(compose, "uOrigin"),
            size: gl.get_uniform_location(compose, "uSize"),
            pixel: gl.get_uniform_location(compose, "uPixel"),
            cell_count: gl.get_uniform_location(compose, "uCellCount"),
            strength: gl.get_uniform_location(compose, "uStrength"),
            light: gl.get_uniform_location(compose, "uLight"),
            ink: gl.get_uniform_location(compose, "uInk"),
            accent: gl.get_uniform_location(compose, "uAccent"),
        };
        Ok(Gpu {
            field,
            field_u,
            compose,
            compose_u,
            vao,
            texture,
            framebuffer,
            cells: (1, 1),
        })
    }
}

const VERTEX: &str = r#"
void main() {
    vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
"#;

/// Pass 1, one fragment per dither cell: r = field value / 1.25, g = cursor/ripple light.
const FIELD: &str = r#"
uniform vec2 uSize;
uniform float uScale;
uniform float uTime;
uniform float uPixel;
uniform float uRows;
uniform float uDensity;
uniform float uVariant;
uniform vec3 uTrail[16];
uniform vec3 uRipples[6];
out vec4 fragColor;

float hash(vec3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.zyx + 31.32);
    return fract((p.x + p.y) * p.z);
}

float noise(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    vec3 w = f * f * (3.0 - 2.0 * f);
    float a = mix(mix(hash(i), hash(i + vec3(1, 0, 0)), w.x),
                  mix(hash(i + vec3(0, 1, 0)), hash(i + vec3(1, 1, 0)), w.x), w.y);
    float b = mix(mix(hash(i + vec3(0, 0, 1)), hash(i + vec3(1, 0, 1)), w.x),
                  mix(hash(i + vec3(0, 1, 1)), hash(i + vec3(1, 1, 1)), w.x), w.y);
    return mix(a, b, w.z);
}

float clouds(vec2 uv, float t) {
    vec3 p = vec3(uv, t);
    float sum = 0.0;
    float amp = 0.55;
    for (int i = 0; i < 4; i++) {
        sum += amp * noise(p);
        p = vec3(p.xy * 2.03 + vec2(17.1, 9.3), p.z * 1.21);
        amp *= 0.5;
    }
    return sum;
}

void main() {
    vec2 cell = vec2(floor(gl_FragCoord.x), uRows - 1.0 - floor(gl_FragCoord.y));
    vec2 center = (cell + 0.5) * uPixel / uScale;
    float aspect = uSize.y / uScale;
    vec2 uv = center / aspect;

    float field;
    if (uVariant < 0.5) {
        vec2 drift = vec2(uTime * 0.035, -uTime * 0.018);
        field = clouds(uv * 2.4 + drift, uTime * 0.05);
        field = smoothstep(0.36, 0.74, field) * (0.55 + uDensity * 0.7) - 0.05;
    } else {
        vec2 c = vec2(uSize.x / uScale * 0.72, uSize.y / uScale * 0.62) / aspect;
        vec2 d = uv - c;
        float a = atan(d.y, d.x);
        float petals = 0.5 + 0.5 * sin(a * 5.0 + uTime * 0.25 + clouds(uv * 1.5, uTime * 0.04) * 3.0);
        float r = length(d) / (0.42 + 0.08 * petals + 0.03 * sin(uTime * 0.4));
        field = (1.0 - smoothstep(0.15, 1.0, r)) * (0.55 + 0.45 * petals);
        field *= 0.75 + uDensity * 0.5;
    }

    float heat = 0.0;
    for (int i = 0; i < 16; i++) {
        vec3 a = uTrail[i];
        if (a.z >= 1.0) continue;
        vec3 b = uTrail[min(i + 1, 15)];
        if (b.z >= 1.0) b = a;
        vec2 ab = b.xy - a.xy;
        float h = clamp(dot(center - a.xy, ab) / max(dot(ab, ab), 1e-3), 0.0, 1.0);
        float d = distance(center, a.xy + ab * h) / 64.0;
        float life = 1.0 - mix(a.z, b.z, h);
        heat = max(heat, exp(-d * d * 1.6) * life);
    }

    float ring = 0.0;
    for (int i = 0; i < 6; i++) {
        vec3 c = uRipples[i];
        if (c.z < 0.0) continue;
        float radius = c.z * 260.0;
        float r = distance(center, c.xy);
        float band = (r - radius) / (26.0 + c.z * 30.0);
        ring = max(ring, 1.2 * exp(-band * band) * exp(-c.z * 1.1));
        ring = max(ring, (1.0 - smoothstep(0.0, 40.0 + c.z * 30.0, r)) * exp(-c.z * 3.5));
    }
    float lit = clamp(max(heat, ring), 0.0, 1.0);
    float value = clamp(field + lit * 0.95, 0.0, 1.25);
    fragColor = vec4(value / 1.25, lit, 0.0, 1.0);
}
"#;

/// Pass 2, full resolution: ordered dither threshold, square pixel mask, vignette, color.
const COMPOSE: &str = r#"
uniform sampler2D uCells;
uniform vec2 uOrigin;
uniform vec2 uSize;
uniform float uPixel;
uniform ivec2 uCellCount;
uniform float uStrength;
uniform float uLight;
uniform vec3 uInk;
uniform vec3 uAccent;
out vec4 fragColor;

float bayer2(vec2 a) { a = floor(a); return fract(a.x * 0.5 + a.y * a.y * 0.75); }
float bayer4(vec2 a) { return bayer2(0.5 * a) * 0.25 + bayer2(a); }
float bayer8(vec2 a) { return bayer4(0.5 * a) * 0.25 + bayer2(a); }

void main() {
    vec2 px = gl_FragCoord.xy - uOrigin;
    px.y = uSize.y - px.y;
    vec2 cell = floor(px / uPixel);
    vec2 local = fract(px / uPixel);
    ivec2 texel = clamp(ivec2(int(cell.x), uCellCount.y - 1 - int(cell.y)), ivec2(0), uCellCount - 1);
    vec4 cellData = texelFetch(uCells, texel, 0);
    float value = cellData.r * 1.25;
    float lit = cellData.g;

    float on = step(bayer8(cell), max(value, lit * 1.25));
    vec2 q = abs(local - 0.5);
    float dotMask = step(max(q.x, q.y), 0.30 + 0.08 * smoothstep(0.2, 0.9, lit));

    vec2 n = (gl_FragCoord.xy - uOrigin) / uSize;
    vec2 edge = smoothstep(vec2(0.0), vec2(0.08), n) * smoothstep(vec2(0.0), vec2(0.08), 1.0 - n);
    float fade = edge.x * edge.y;

    float tone = smoothstep(0.0, 0.85, lit);
    vec3 color = mix(uInk, uAccent, tone);
    float alpha = on * dotMask * fade * mix(0.30 + 0.22 * uLight, 0.95, tone) * uStrength;
    fragColor = vec4(color * alpha, alpha);
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pointer_history_is_bounded_and_expires() {
        let mut input = DitherInput::default();
        for i in 0..40 {
            input.record(
                f64::from(i) * 0.01,
                Some(egui::pos2(i as f32 * 10.0, 0.0)),
                i % 5 == 0,
            );
        }
        assert!(input.trail.len() <= TRAIL);
        assert!(input.ripples.len() <= RIPPLES);
        input.record(10.0, None, false);
        assert!(!input.busy());

        let mut frame = DitherFrame::default();
        input.record(10.0, Some(egui::pos2(50.0, 60.0)), true);
        input.fill(
            &mut frame,
            egui::Rect::from_min_size(egui::pos2(10.0, 20.0), egui::vec2(100.0, 100.0)),
            10.2,
        );
        assert_eq!(frame.trail[0][0], 40.0);
        assert!(frame.trail[1][2] >= 1.0);
        assert!((frame.ripples[0][2] - 0.2).abs() < 1e-4);
        assert!(frame.ripples[1][2] < 0.0);
    }

    #[test]
    fn clear_empties_trail_and_ripples() {
        let mut input = DitherInput::default();
        input.record(1.0, Some(egui::pos2(20.0, 30.0)), true);
        assert!(input.busy());
        input.clear();
        assert!(!input.busy());
    }
}

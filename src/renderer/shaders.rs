//! User post-processing shaders, written in SkSL and run over the whole rendered frame.

use std::{
    path::PathBuf,
    time::{Duration, Instant, SystemTime},
};

use itertools::Itertools;
use log::error;
use rmpv::Value;
use skia_safe::{
    BlendMode, Canvas, Color4f, Data, FilterMode, ISize, Image, ImageInfo, Matrix, MipmapMode,
    Paint, Rect, RuntimeEffect, SamplingOptions, Shader, Surface, SurfaceProps, TileMode,
    canvas::SaveLayerRec,
    image_filters,
    runtime_effect::{ChildType, RuntimeShaderBuilder, uniform},
};
use winit::event::WindowEvent;

use crate::{error_msg, settings::ParseFromValue};

/// The child shader a shader reads the rendered frame from.
const FRAME: &str = "frame";
/// Tells Skia that shaders read `frame` around the pixel they compute, not only at it.
const SAMPLE_RADIUS: f32 = 32.0;
/// How often the shader files are checked for changes.
const RELOAD_CHECK_INTERVAL: Duration = Duration::from_millis(500);

/// The uniforms a shader can declare, with their types.
const UNIFORMS: &[(&str, uniform::Type)] = &[
    ("resolution", uniform::Type::Float2),
    ("time", uniform::Type::Float),
    ("time_delta", uniform::Type::Float),
    ("cursor", uniform::Type::Float4),
    ("cursor_color", uniform::Type::Float4),
    ("mouse", uniform::Type::Float2),
    ("focused", uniform::Type::Float),
];

/// The `neovide_shader` setting: the shader files applied to every frame, in order.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ShaderPaths(Vec<String>);

impl ShaderPaths {
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl ParseFromValue for ShaderPaths {
    fn parse_from_value(&mut self, value: Value) {
        if let Some(path) = value.as_str() {
            self.0 =
                Some(path).filter(|path| !path.is_empty()).map(String::from).into_iter().collect();
        } else if let Some(items) = value.as_array() {
            self.0 = items
                .iter()
                .filter_map(|item| {
                    let path = item.as_str();
                    if path.is_none() {
                        error!("Expected a shader path string in the array, but received {item:?}");
                    }
                    path
                })
                .filter(|path| !path.is_empty())
                .map(String::from)
                .collect();
        } else {
            error!("Expected a shader path or an array of shader paths, but received {value:?}");
        }
    }
}

impl From<ShaderPaths> for Value {
    fn from(paths: ShaderPaths) -> Self {
        match paths.0.as_slice() {
            [] => Value::from(""),
            [path] => Value::from(path.as_str()),
            paths => {
                Value::from(paths.iter().map(|path| Value::from(path.as_str())).collect::<Vec<_>>())
            }
        }
    }
}

fn expand_path(path: &str) -> PathBuf {
    match (path.strip_prefix('~'), dirs::home_dir()) {
        (Some(rest), Some(home)) if rest.is_empty() || rest.starts_with(['/', '\\']) => {
            home.join(rest.trim_start_matches(['/', '\\']))
        }
        _ => PathBuf::from(path),
    }
}

/// A shader that compiled and only uses inputs Neovide provides.
struct CompiledShader {
    effect: RuntimeEffect,
    reads_frame: bool,
    animated: bool,
}

fn type_name(ty: uniform::Type) -> &'static str {
    match ty {
        uniform::Type::Float => "float",
        uniform::Type::Float2 => "float2",
        uniform::Type::Float4 => "float4 or half4",
        _ => "?",
    }
}

fn supported_uniforms() -> String {
    UNIFORMS.iter().map(|(name, ty)| format!("{} {name}", type_name(*ty))).join(", ")
}

fn compile(source: &str) -> Result<CompiledShader, String> {
    let effect = RuntimeEffect::make_for_shader(source, None)?;

    for uniform in effect.uniforms() {
        let name = uniform.name();
        match UNIFORMS.iter().find(|(known, _)| *known == name) {
            None => {
                return Err(format!(
                    "unknown uniform `{name}`, the supported uniforms are: {}",
                    supported_uniforms()
                ));
            }
            Some((_, ty)) if *ty != uniform.ty() || uniform.is_array() => {
                return Err(format!("uniform `{name}` has to be a {}", type_name(*ty)));
            }
            Some(_) => {}
        }
    }

    for child in effect.children() {
        if child.name() != FRAME || child.ty() != ChildType::Shader {
            return Err(format!(
                "unknown child `{}`, the rendered frame is `uniform shader {FRAME}`",
                child.name()
            ));
        }
    }

    Ok(CompiledShader {
        reads_frame: effect.find_child(FRAME).is_some(),
        animated: effect.find_uniform("time").is_some()
            || effect.find_uniform("time_delta").is_some(),
        effect,
    })
}

/// One configured shader file.
struct ShaderSlot {
    path: PathBuf,
    checked: bool,
    /// Modification time of the file when it was last loaded, or when loading it last failed.
    modified: Option<SystemTime>,
    loaded_at: Instant,
    shader: Option<CompiledShader>,
    /// Whether the shader failed to run since it was loaded, so it's reported once, not every frame.
    failure_reported: bool,
}

impl ShaderSlot {
    fn new(path: PathBuf) -> Self {
        Self {
            path,
            checked: false,
            modified: None,
            loaded_at: Instant::now(),
            shader: None,
            failure_reported: false,
        }
    }

    fn report_failure(&mut self) {
        if !self.failure_reported {
            self.failure_reported = true;
            error!("Shader {} couldn't be run", self.path.display());
        }
    }

    /// Loads the file when it's new or changed, and returns whether a new version was loaded.
    /// A failed reload keeps the previous shader.
    fn reload_if_changed(&mut self, report_error: fn(&str)) -> bool {
        let modified = std::fs::metadata(&self.path).and_then(|metadata| metadata.modified()).ok();
        if self.checked && modified == self.modified {
            return false;
        }
        self.checked = true;
        self.modified = modified;

        let shader = std::fs::read_to_string(&self.path)
            .map_err(|error| error.to_string())
            .and_then(|source| compile(&source));
        match shader {
            Ok(shader) => {
                log::info!("Loaded shader {}", self.path.display());
                self.shader = Some(shader);
                self.loaded_at = Instant::now();
                self.failure_reported = false;
                true
            }
            Err(error) => {
                report_error(&format!("Shader {}: {error}", self.path.display()));
                false
            }
        }
    }
}

/// What the renderer knows about the current frame, for the shader uniforms.
pub struct FrameInputs {
    pub time_delta: f32,
    pub cursor: Rect,
    /// Premultiplied.
    pub cursor_color: Color4f,
}

/// Offscreen surfaces for the window's exact image info and surface props: the frame is drawn into
/// the first one, and when shaders are chained, each one but the last draws into the next one.
struct Surfaces {
    key: (ImageInfo, SurfaceProps),
    stages: Vec<Surface>,
}

/// Runs the shaders from the `neovide_shader` setting over everything that was drawn.
pub struct PostProcessor {
    configured: ShaderPaths,
    slots: Vec<ShaderSlot>,
    last_check: Option<Instant>,
    focused: bool,
    mouse: (f32, f32),
    report_error: fn(&str),
    surfaces: Option<Surfaces>,
}

fn show_error(message: &str) {
    error_msg!("{message}");
}

impl Default for PostProcessor {
    fn default() -> Self {
        Self {
            configured: ShaderPaths::default(),
            slots: Vec::new(),
            last_check: None,
            focused: true,
            mouse: (0.0, 0.0),
            report_error: show_error,
            surfaces: None,
        }
    }
}

impl PostProcessor {
    /// Follows the setting, and reloads shader files that were saved since they were loaded.
    /// Returns whether the shaders changed, so the frame needs redrawing.
    pub fn sync(&mut self, paths: &ShaderPaths) -> bool {
        if paths.is_empty() {
            let had_shaders = !self.slots.is_empty();
            self.slots.clear();
            self.configured = ShaderPaths::default();
            self.surfaces = None;
            return had_shaders;
        }

        let mut changed = false;
        if *paths != self.configured {
            self.configured = paths.clone();
            self.slots = paths.0.iter().map(|path| ShaderSlot::new(expand_path(path))).collect();
            self.last_check = None;
            changed = true;
        }

        if self.last_check.is_some_and(|last_check| last_check.elapsed() < RELOAD_CHECK_INTERVAL) {
            return changed;
        }
        self.last_check = Some(Instant::now());
        for slot in &mut self.slots {
            changed |= slot.reload_if_changed(self.report_error);
        }
        changed
    }

    fn shaders(&self) -> impl Iterator<Item = (&ShaderSlot, &CompiledShader)> {
        self.slots.iter().filter_map(|slot| slot.shader.as_ref().map(|shader| (slot, shader)))
    }

    pub fn has_shaders(&self) -> bool {
        self.shaders().next().is_some()
    }

    /// Shaders that use time keep redrawing, but only while the window has focus.
    pub fn is_animating(&self) -> bool {
        self.focused && self.shaders().any(|(_, shader)| shader.animated)
    }

    pub fn handle_event(&mut self, event: &WindowEvent) {
        match event {
            WindowEvent::Focused(focused) => self.focused = *focused,
            WindowEvent::CursorMoved { position, .. } => {
                self.mouse = (position.x as f32, position.y as f32);
            }
            _ => {}
        }
    }

    /// The value of a uniform for this frame, and how many of its floats are used.
    fn uniform_value(
        &self,
        name: &str,
        loaded_at: Instant,
        size: ISize,
        inputs: &FrameInputs,
    ) -> Option<([f32; 4], usize)> {
        let (cursor, color) = (inputs.cursor, inputs.cursor_color);
        Some(match name {
            "resolution" => ([size.width as f32, size.height as f32, 0.0, 0.0], 2),
            "time" => ([loaded_at.elapsed().as_secs_f32(), 0.0, 0.0, 0.0], 1),
            "time_delta" => ([inputs.time_delta, 0.0, 0.0, 0.0], 1),
            "cursor" => ([cursor.left, cursor.top, cursor.width(), cursor.height()], 4),
            "cursor_color" => ([color.r, color.g, color.b, color.a], 4),
            "mouse" => ([self.mouse.0, self.mouse.1, 0.0, 0.0], 2),
            "focused" => ([if self.focused { 1.0 } else { 0.0 }, 0.0, 0.0, 0.0], 1),
            _ => return None,
        })
    }

    /// The shader of the slot, reading `frame` from `image`. Failures are reported once.
    fn shader_output(
        &mut self,
        slot_index: usize,
        image: &Image,
        size: ISize,
        inputs: &FrameInputs,
    ) -> Option<Shader> {
        let slot = &self.slots[slot_index];
        let compiled = slot.shader.as_ref()?;
        let effect = &compiled.effect;
        let mut uniforms = vec![0u8; effect.uniform_size()];
        for uniform in effect.uniforms() {
            let Some((value, len)) =
                self.uniform_value(uniform.name(), slot.loaded_at, size, inputs)
            else {
                continue;
            };
            let bytes: Vec<u8> = value[..len].iter().flat_map(|v| v.to_ne_bytes()).collect();
            let offset = uniform.offset();
            uniforms[offset..offset + bytes.len()].copy_from_slice(&bytes);
        }
        let children = if compiled.reads_frame {
            // Linear filtering like the image filter fallback, and clamped at the window edges
            let sampling = SamplingOptions::new(FilterMode::Linear, MipmapMode::None);
            image
                .to_shader(Some((TileMode::Clamp, TileMode::Clamp)), sampling, None)
                .map(Into::into)
        } else {
            None
        };
        let output = effect.make_shader(Data::new_copy(&uniforms), children.as_slice(), None);
        if output.is_none() {
            self.slots[slot_index].report_failure();
        }
        output
    }

    /// Starts a frame. While shaders are active this returns the offscreen surface to draw the frame
    /// into, which `present` then runs the shaders over. `None` means drawing into the window.
    pub fn begin_frame(&mut self, window: &Canvas) -> Option<Surface> {
        if !self.has_shaders() {
            self.surfaces = None;
            return None;
        }
        let key = (window.image_info(), window.base_props());
        if self.surfaces.as_ref().is_none_or(|surfaces| surfaces.key != key) {
            self.surfaces = None;
            let surface = window.new_surface(&key.0, Some(&key.1))?;
            log::debug!("Created the shader frame surface, {:?}", key.0.dimensions());
            self.surfaces = Some(Surfaces { key, stages: vec![surface] });
        }
        let mut frame = self.surfaces.as_ref()?.stages[0].clone();
        // A canvas reused across frames keeps any save it was left with
        frame.canvas().restore_to_count(1);
        Some(frame)
    }

    /// The surface for the output of the shader at `stage` in a chain, created on first use.
    fn stage_surface(&mut self, stage: usize, window: &Canvas) -> Option<Surface> {
        let surfaces = self.surfaces.as_mut()?;
        while surfaces.stages.len() <= stage {
            surfaces.stages.push(window.new_surface(&surfaces.key.0, Some(&surfaces.key.1))?);
        }
        Some(surfaces.stages[stage].clone())
    }

    /// Runs the shaders over the frame drawn into `frame` and draws the result into the window.
    /// Without an offscreen frame, it filters what was drawn into the window instead.
    pub fn present(&mut self, frame: Option<Surface>, window: &Canvas, inputs: &FrameInputs) {
        let Some(frame) = frame else {
            self.apply(window, inputs);
            return;
        };
        let size = window.base_layer_size();
        let bounds = Rect::from_wh(size.width as f32, size.height as f32);
        let active: Vec<usize> =
            (0..self.slots.len()).filter(|&index| self.slots[index].shader.is_some()).collect();

        let mut source = frame;
        let mut stage = 0;
        let mut on_window = false;
        for (position, &slot_index) in active.iter().enumerate() {
            let last = position + 1 == active.len();
            let output = {
                // Dropped before anything is drawn into this surface again, so it's never copied
                let image =
                    source.make_temporary_image().unwrap_or_else(|| source.image_snapshot());
                self.shader_output(slot_index, &image, size, inputs)
            };
            let Some(output) = output else {
                continue;
            };
            let paint =
                Paint::default().set_shader(output).set_blend_mode(BlendMode::Src).to_owned();
            if last {
                window.draw_rect(bounds, &paint);
                on_window = true;
            } else {
                // Each stage writes its own surface, so nothing is overwritten after being read
                let Some(mut target) = self.stage_surface(stage + 1, window) else {
                    break;
                };
                target.canvas().draw_rect(bounds, &paint);
                stage += 1;
                source = target;
            }
        }
        if !on_window {
            // A shader failed, so show the frame as far as the shaders got
            let paint = Paint::default().set_blend_mode(BlendMode::Src).to_owned();
            source.draw(window, (0, 0), SamplingOptions::default(), Some(&paint));
        }
    }

    /// Releases the GPU surfaces, which must happen before their Skia context goes away.
    pub fn release_gpu_resources(&mut self) {
        self.surfaces = None;
    }

    /// The fallback when no offscreen surface could be made: replaces what was drawn into the
    /// canvas with the output of each shader in turn.
    pub fn apply(&mut self, canvas: &Canvas, inputs: &FrameInputs) {
        let size = canvas.base_layer_size();
        for slot_index in 0..self.slots.len() {
            let slot = &self.slots[slot_index];
            let Some(shader) = slot.shader.as_ref() else {
                continue;
            };
            let mut builder = RuntimeShaderBuilder::new(shader.effect.clone());
            for uniform in shader.effect.uniforms() {
                let name = uniform.name();
                if let Some((value, len)) = self.uniform_value(name, slot.loaded_at, size, inputs)
                    && let Err(error) = builder.set_uniform_float(name, &value[..len])
                {
                    error!("Couldn't set shader uniform `{name}`: {error:?}");
                }
            }

            let filter = if shader.reads_frame {
                image_filters::runtime_shader_with_options(
                    &builder,
                    SAMPLE_RADIUS,
                    FRAME,
                    None,
                    false,
                )
            } else {
                builder
                    .make_shader(&Matrix::default())
                    .and_then(|output| image_filters::shader(output, None))
            };
            let Some(filter) = filter else {
                self.slots[slot_index].report_failure();
                continue;
            };

            // The shader output replaces the frame, alpha included, so transparency is up to it
            let paint = Paint::default().set_blend_mode(BlendMode::Src).to_owned();
            canvas.save_layer(&SaveLayerRec::default().backdrop(&filter).paint(&paint));
            canvas.restore();
        }
    }
}

#[cfg(test)]
mod tests {
    use skia_safe::{AlphaType, ColorType, ImageInfo, Surface, surfaces};

    use super::*;

    const ALL_UNIFORMS: &str = "
        uniform shader frame;
        uniform float2 resolution;
        uniform float time;
        uniform float time_delta;
        uniform float4 cursor;
        uniform half4 cursor_color;
        uniform float2 mouse;
        uniform float focused;
        half4 main(float2 coord) {
            float s = time + time_delta + focused + resolution.x + mouse.x + cursor.x;
            return frame.eval(coord) * half(step(0.0, s)) + cursor_color * 0.0;
        }
    ";

    fn post_processor(sources: &[&str]) -> PostProcessor {
        let slots = sources
            .iter()
            .map(|source| ShaderSlot {
                shader: Some(compile(source).unwrap()),
                ..ShaderSlot::new(PathBuf::from("test.sksl"))
            })
            .collect();
        PostProcessor { slots, ..Default::default() }
    }

    fn frame_inputs() -> FrameInputs {
        FrameInputs {
            time_delta: 0.0,
            cursor: Rect::new_empty(),
            cursor_color: Color4f::new(0.0, 0.0, 0.0, 0.0),
        }
    }

    fn pixels(surface: &mut Surface) -> Vec<[u8; 4]> {
        let size = surface.image_info().dimensions();
        let info = ImageInfo::new(size, ColorType::RGBA8888, AlphaType::Premul, None);
        let mut bytes = vec![0; (size.width * size.height * 4) as usize];
        assert!(surface.read_pixels(&info, &mut bytes, size.width as usize * 4, (0, 0)));
        bytes.chunks(4).map(|pixel| [pixel[0], pixel[1], pixel[2], pixel[3]]).collect()
    }

    /// Draws a frame the way the renderer does: into the offscreen frame surface while shaders are
    /// active, then through the shaders into the window.
    fn render(post_processor: &mut PostProcessor, window: &mut Surface, scene: impl Fn(&Canvas)) {
        let mut frame = post_processor.begin_frame(window.canvas());
        match frame.as_mut() {
            Some(frame) => scene(frame.canvas()),
            None => scene(window.canvas()),
        }
        post_processor.present(frame, window.canvas(), &frame_inputs());
    }

    /// Runs the shaders over a 4x1 window, both through the offscreen frame and through the
    /// fallback that filters the window directly, which must agree.
    fn run(sources: &[&str], fill: impl Fn(&Canvas)) -> Vec<[u8; 4]> {
        let mut window = surfaces::raster_n32_premul((4, 1)).unwrap();
        render(&mut post_processor(sources), &mut window, &fill);
        let offscreen = pixels(&mut window);

        let mut window = surfaces::raster_n32_premul((4, 1)).unwrap();
        fill(window.canvas());
        post_processor(sources).apply(window.canvas(), &frame_inputs());
        assert_eq!(offscreen, pixels(&mut window), "offscreen and fallback paths differ");
        offscreen
    }

    /// The offscreen frame surface, kept alive so its address can't be reused, and its canvas,
    /// which identifies it.
    fn frame_surface(post_processor: &PostProcessor) -> (Surface, *const Canvas) {
        let mut surface = post_processor.surfaces.as_ref().unwrap().stages[0].clone();
        let canvas: *const Canvas = surface.canvas();
        (surface, canvas)
    }

    fn fill_rgb(canvas: &Canvas, r: f32, g: f32, b: f32) {
        canvas.clear(Color4f::new(r, g, b, 1.0));
    }

    fn close(actual: [u8; 4], expected: [u8; 4]) -> bool {
        actual.iter().zip(expected).all(|(a, e)| a.abs_diff(e) <= 1)
    }

    #[test]
    fn shader_paths_parse_a_path_a_list_or_nothing() {
        let mut paths = ShaderPaths::default();
        paths.parse_from_value(Value::from("~/a.sksl"));
        assert_eq!(paths, ShaderPaths(vec!["~/a.sksl".into()]));

        paths.parse_from_value(Value::from(vec![Value::from("a.sksl"), Value::from("b.sksl")]));
        assert_eq!(paths, ShaderPaths(vec!["a.sksl".into(), "b.sksl".into()]));

        paths.parse_from_value(Value::from(""));
        assert!(paths.is_empty());
        assert_eq!(Value::from(paths), Value::from(""));
    }

    #[test]
    fn expand_path_resolves_home() {
        let home = dirs::home_dir().unwrap();
        assert_eq!(expand_path("~/shaders/a.sksl"), home.join("shaders/a.sksl"));
        assert_eq!(expand_path("/tmp/a.sksl"), PathBuf::from("/tmp/a.sksl"));
    }

    #[test]
    fn compile_accepts_every_supported_uniform() {
        let shader = compile(ALL_UNIFORMS).unwrap();
        assert!(shader.reads_frame);
        assert!(shader.animated);
    }

    #[test]
    fn compile_rejects_unknown_and_mistyped_uniforms() {
        let error = compile("uniform float speed; half4 main(float2 c) { return half4(speed); }")
            .err()
            .unwrap();
        assert!(error.contains("unknown uniform `speed`"), "{error}");
        assert!(error.contains("float2 resolution"), "{error}");

        let error =
            compile("uniform float2 time; half4 main(float2 c) { return half4(time, 0, 1); }")
                .err()
                .unwrap();
        assert!(error.contains("uniform `time` has to be a float"), "{error}");

        let error = compile("half4 main(float2 c) { return half4(1) }").err().unwrap();
        assert!(!error.is_empty());
    }

    #[test]
    fn invert_shader_inverts_the_frame() {
        let invert = "uniform shader frame;
            half4 main(float2 coord) { half4 c = frame.eval(coord); return half4(c.a - c.rgb, c.a); }";
        let pixels = run(&[invert], |canvas| fill_rgb(canvas, 0.2, 0.4, 0.6));
        assert!(close(pixels[0], [204, 153, 102, 255]), "{pixels:?}");
    }

    #[test]
    fn passthrough_shader_keeps_transparency() {
        let passthrough =
            "uniform shader frame; half4 main(float2 coord) { return frame.eval(coord); }";
        let pixels = run(&[passthrough], |canvas| {
            canvas.clear(Color4f::new(1.0, 0.0, 0.0, 0.5));
        });
        assert!(close(pixels[0], [128, 0, 0, 128]), "{pixels:?}");
    }

    #[test]
    fn frame_is_sampled_in_window_pixels() {
        let shift = "uniform shader frame;
            half4 main(float2 coord) { return frame.eval(coord + float2(1, 0)); }";
        let pixels = run(&[shift], |canvas| {
            fill_rgb(canvas, 1.0, 0.0, 0.0);
            let blue = Paint::new(Color4f::new(0.0, 0.0, 1.0, 1.0), None);
            canvas.draw_rect(Rect::new(2.0, 0.0, 4.0, 1.0), &blue);
        });
        // Each pixel shows its right neighbour, so the red/blue boundary moves one pixel left
        assert!(close(pixels[0], [255, 0, 0, 255]), "{pixels:?}");
        assert!(close(pixels[1], [0, 0, 255, 255]), "{pixels:?}");
        assert!(close(pixels[2], [0, 0, 255, 255]), "{pixels:?}");
    }

    #[test]
    fn shaders_apply_in_order() {
        let invert = "uniform shader frame;
            half4 main(float2 coord) { half4 c = frame.eval(coord); return half4(c.a - c.rgb, c.a); }";
        let red_only = "uniform shader frame;
            half4 main(float2 coord) { return half4(frame.eval(coord).r, 0, 0, 1); }";

        let pixels = run(&[invert, red_only], |canvas| fill_rgb(canvas, 0.2, 0.4, 0.6));
        assert!(close(pixels[0], [204, 0, 0, 255]), "{pixels:?}");

        let pixels = run(&[red_only, invert], |canvas| fill_rgb(canvas, 0.2, 0.4, 0.6));
        assert!(close(pixels[0], [204, 255, 255, 255]), "{pixels:?}");
    }

    #[test]
    fn time_based_shaders_animate_only_while_focused() {
        let pulse = "uniform shader frame; uniform float time;
            half4 main(float2 coord) { return frame.eval(coord) * half(fract(time)); }";
        let mut post_processor = post_processor(&[pulse]);
        assert!(post_processor.is_animating());

        post_processor.handle_event(&WindowEvent::Focused(false));
        assert!(!post_processor.is_animating());

        let still = "uniform shader frame; half4 main(float2 coord) { return frame.eval(coord); }";
        assert!(!self::post_processor(&[still]).is_animating());
    }

    #[test]
    fn missing_file_is_reported_once() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static REPORTS: AtomicUsize = AtomicUsize::new(0);

        let mut post_processor = PostProcessor {
            report_error: |_| {
                REPORTS.fetch_add(1, Ordering::SeqCst);
            },
            ..Default::default()
        };
        let mut paths = ShaderPaths::default();
        paths.parse_from_value(Value::from("/nonexistent/neovide-test-shader.sksl"));

        post_processor.sync(&paths);
        post_processor.last_check = None;
        post_processor.sync(&paths);

        assert_eq!(REPORTS.load(Ordering::SeqCst), 1);
        assert!(!post_processor.has_shaders());
    }

    #[test]
    fn frame_can_be_read_far_from_the_pixel() {
        let shift = "uniform shader frame;
            half4 main(float2 coord) { return frame.eval(coord + float2(60, 0)); }";
        let mut surface = surfaces::raster_n32_premul((100, 1)).unwrap();
        render(&mut post_processor(&[shift]), &mut surface, |canvas| {
            fill_rgb(canvas, 1.0, 0.0, 0.0);
            let blue = Paint::new(Color4f::new(0.0, 0.0, 1.0, 1.0), None);
            canvas.draw_rect(Rect::new(70.0, 0.0, 100.0, 1.0), &blue);
        });

        let pixels = pixels(&mut surface);
        assert!(close(pixels[0], [255, 0, 0, 255]), "{:?}", pixels[0]);
        assert!(close(pixels[15], [0, 0, 255, 255]), "{:?}", pixels[15]);
    }

    #[test]
    fn documented_examples_compile() {
        let docs = include_str!("../../website/docs/configuration.md");
        let section =
            &docs[docs.find("#### Shaders").unwrap()..docs.find("### Functionality").unwrap()];
        let examples =
            section.split("```glsl").skip(1).map(|block| block.split("```").next().unwrap());
        let mut count = 0;
        for example in examples {
            compile(example).unwrap_or_else(|error| panic!("{error}\n{example}"));
            count += 1;
        }
        assert_eq!(count, 2);
    }

    /// Runs the passes on the Metal GPU backend Neovide uses on macOS, not just the CPU one.
    #[cfg(target_os = "macos")]
    #[test]
    fn shaders_run_on_metal() {
        use objc2::rc::Retained;
        use objc2_metal::{MTLCreateSystemDefaultDevice, MTLDevice};
        use skia_safe::gpu::{self, Budgeted, SurfaceOrigin, mtl::BackendContext};

        let device = MTLCreateSystemDefaultDevice().expect("no Metal device");
        let queue = device.newCommandQueue().unwrap();
        let backend = unsafe {
            BackendContext::new(Retained::as_ptr(&device).cast(), Retained::as_ptr(&queue).cast())
        };
        let mut context = gpu::direct_contexts::make_metal(&backend, None).unwrap();
        // Like the window surface: BGRA, top-left origin, one sample
        let info = ImageInfo::new((4, 2), ColorType::BGRA8888, AlphaType::Premul, None);
        let mut surface = gpu::surfaces::render_target(
            &mut context,
            Budgeted::Yes,
            &info,
            1,
            SurfaceOrigin::TopLeft,
            None,
            false,
            None,
        )
        .unwrap();

        let scene = |canvas: &Canvas| {
            fill_rgb(canvas, 1.0, 0.0, 0.0);
            let blue = Paint::new(Color4f::new(0.0, 0.0, 1.0, 1.0), None);
            canvas.draw_rect(Rect::new(2.0, 0.0, 4.0, 2.0), &blue);
        };
        let shift = "uniform shader frame;
            half4 main(float2 coord) { return frame.eval(coord + float2(1, 0)); }";
        let top_green_else_invert = "uniform shader frame; uniform float2 resolution;
            half4 main(float2 coord) {
                half4 c = frame.eval(coord);
                return coord.y < resolution.y * 0.5 ? half4(0, 1, 0, 1) : half4(c.a - c.rgb, c.a);
            }";
        for offscreen in [true, false] {
            let mut post_processor = post_processor(&[shift, top_green_else_invert]);
            if offscreen {
                render(&mut post_processor, &mut surface, scene);
            } else {
                scene(surface.canvas());
                post_processor.apply(surface.canvas(), &frame_inputs());
            }
            context.flush_and_submit();

            let pixels = pixels(&mut surface);
            // Top row is green, the bottom row is shifted one pixel left, then inverted
            for pixel in &pixels[0..4] {
                assert!(close(*pixel, [0, 255, 0, 255]), "{pixels:?}");
            }
            assert!(close(pixels[4], [0, 255, 255, 255]), "{pixels:?}");
            assert!(close(pixels[5], [255, 255, 0, 255]), "{pixels:?}");
            assert!(close(pixels[6], [255, 255, 0, 255]), "{pixels:?}");
        }
    }

    /// Like the window: a Metal texture wrapped as a backend render target, with text properties.
    #[cfg(target_os = "macos")]
    #[test]
    fn offscreen_frame_matches_direct_rendering_on_a_metal_window_target() {
        use objc2::rc::Retained;
        use objc2_metal::{
            MTLCreateSystemDefaultDevice, MTLDevice, MTLPixelFormat, MTLTextureDescriptor,
            MTLTextureUsage,
        };
        use skia_safe::{
            ColorSpace, Font, FontMgr, FontStyle, PixelGeometry, Point, SurfacePropsFlags,
            gpu::{
                self, SurfaceOrigin, backend_render_targets,
                mtl::{BackendContext, TextureInfo},
                surfaces::wrap_backend_render_target,
            },
        };

        let device = MTLCreateSystemDefaultDevice().expect("no Metal device");
        let queue = device.newCommandQueue().unwrap();
        let backend = unsafe {
            BackendContext::new(Retained::as_ptr(&device).cast(), Retained::as_ptr(&queue).cast())
        };
        let mut context = gpu::direct_contexts::make_metal(&backend, None).unwrap();
        let descriptor = unsafe {
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                MTLPixelFormat::BGRA8Unorm,
                160,
                40,
                false,
            )
        };
        descriptor.setUsage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead);
        let texture = device.newTextureWithDescriptor(&descriptor).unwrap();
        let texture_info = unsafe { TextureInfo::new(Retained::as_ptr(&texture).cast()) };
        let target = backend_render_targets::make_mtl((160, 40), &texture_info);
        let props = SurfaceProps::new_with_text_properties(
            SurfacePropsFlags::default(),
            PixelGeometry::Unknown,
            1.0,
            3.0,
        );
        let mut window = wrap_backend_render_target(
            &mut context,
            &target,
            SurfaceOrigin::TopLeft,
            ColorType::BGRA8888,
            ColorSpace::new_srgb(),
            Some(&props),
        )
        .unwrap();

        // A translucent background with text, as Neovide draws it
        let typeface = FontMgr::new().legacy_make_typeface(None, FontStyle::normal()).unwrap();
        let font = Font::from_typeface(typeface, 20.0);
        let scene = |canvas: &Canvas| {
            canvas.clear(Color4f::new(0.05, 0.07, 0.08, 0.9));
            let paint = Paint::new(Color4f::new(0.88, 0.86, 0.62, 1.0), None);
            canvas.draw_str("fn main() -> Ok", Point::new(4.0, 28.0), &font, &paint);
        };
        scene(window.canvas());
        context.flush_and_submit();
        let direct = pixels(&mut window);

        let passthrough =
            "uniform shader frame; half4 main(float2 coord) { return frame.eval(coord); }";
        let mut post_processor = post_processor(&[passthrough]);
        render(&mut post_processor, &mut window, scene);
        context.flush_and_submit();

        let surfaces = post_processor.surfaces.as_ref().expect("offscreen surface");
        assert!(surfaces.stages[0].image_info() == window.image_info());
        assert!(surfaces.stages[0].props() == window.props());
        assert!(direct.iter().any(|pixel| pixel[0] > 100), "the scene has no text");
        let shaded = pixels(&mut window);
        let differing = direct.iter().zip(&shaded).filter(|(a, b)| a != b).count();
        assert_eq!(
            differing, 0,
            "pixels where the shaded passthrough differs from direct rendering"
        );
    }

    #[test]
    fn fractional_sampling_is_linear() {
        let quarter = "uniform shader frame;
            half4 main(float2 coord) { return frame.eval(coord + float2(0.25, 0)); }";
        let pixels = run(&[quarter], |canvas| {
            fill_rgb(canvas, 1.0, 0.0, 0.0);
            canvas.draw_rect(
                Rect::new(2.0, 0.0, 4.0, 1.0),
                &Paint::new(Color4f::new(0.0, 0.0, 1.0, 1.0), None),
            );
        });
        // A quarter of the way from the red pixel's center to the blue one's
        assert!(close(pixels[1], [191, 0, 64, 255]), "{pixels:?}");
    }

    #[test]
    fn reads_past_the_window_edge_get_the_edge_pixel() {
        let shift = "uniform shader frame;
            half4 main(float2 coord) { return frame.eval(coord + float2(2, 0)); }";
        let mut window = surfaces::raster_n32_premul((4, 1)).unwrap();
        render(&mut post_processor(&[shift]), &mut window, |canvas| {
            fill_rgb(canvas, 1.0, 0.0, 0.0);
            canvas.draw_rect(
                Rect::new(3.0, 0.0, 4.0, 1.0),
                &Paint::new(Color4f::new(0.0, 0.0, 1.0, 1.0), None),
            );
        });
        let pixels = pixels(&mut window);
        assert!(close(pixels[3], [0, 0, 255, 255]), "{pixels:?}");
    }

    #[test]
    fn failing_shader_still_shows_the_frame() {
        let invert = "uniform shader frame;
            half4 main(float2 coord) { half4 c = frame.eval(coord); return half4(c.a - c.rgb, c.a); }";
        // Claiming not to read the frame leaves out the child the effect needs, so it can't run
        let mut broken = compile(invert).unwrap();
        broken.reads_frame = false;
        let mut chain = post_processor(&[invert]);
        chain.slots.push(ShaderSlot {
            shader: Some(broken),
            ..ShaderSlot::new(PathBuf::from("broken.sksl"))
        });

        let mut window = surfaces::raster_n32_premul((4, 1)).unwrap();
        render(&mut chain, &mut window, |canvas| fill_rgb(canvas, 0.2, 0.4, 0.6));
        let pixels = pixels(&mut window);
        // The first shader still ran, and its result reached the window
        assert!(close(pixels[0], [204, 153, 102, 255]), "{pixels:?}");
        assert!(chain.slots[1].failure_reported);
    }

    #[test]
    fn frame_surface_is_reused_across_frames() {
        let passthrough =
            "uniform shader frame; half4 main(float2 coord) { return frame.eval(coord); }";
        let mut post_processor = post_processor(&[passthrough]);
        let mut window = surfaces::raster_n32_premul((4, 1)).unwrap();

        render(&mut post_processor, &mut window, |canvas| fill_rgb(canvas, 1.0, 0.0, 0.0));
        let (_first, first) = frame_surface(&post_processor);
        render(&mut post_processor, &mut window, |canvas| fill_rgb(canvas, 0.0, 1.0, 0.0));
        assert_eq!(first, frame_surface(&post_processor).1);
        assert!(close(pixels(&mut window)[0], [0, 255, 0, 255]));

        // A different window size needs a new surface, and turning shaders off releases it
        let mut bigger = surfaces::raster_n32_premul((8, 2)).unwrap();
        render(&mut post_processor, &mut bigger, |canvas| fill_rgb(canvas, 0.0, 0.0, 1.0));
        assert_ne!(first, frame_surface(&post_processor).1);
        assert!(close(pixels(&mut bigger)[15], [0, 0, 255, 255]));
        post_processor.sync(&ShaderPaths::default());
        assert!(post_processor.surfaces.is_none());
    }

    #[test]
    fn sync_reports_when_shaders_change() {
        let path =
            std::env::temp_dir().join(format!("neovide-shader-test-{}.sksl", std::process::id()));
        let passthrough =
            "uniform shader frame; half4 main(float2 coord) { return frame.eval(coord); }";
        std::fs::write(&path, passthrough).unwrap();
        let mut paths = ShaderPaths::default();
        paths.parse_from_value(Value::from(path.to_str().unwrap()));
        let mut post_processor = PostProcessor::default();

        assert!(post_processor.sync(&paths), "first load");
        assert!(post_processor.has_shaders());
        post_processor.last_check = None;
        assert!(!post_processor.sync(&paths), "nothing changed");

        let file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        file.set_modified(SystemTime::now() + Duration::from_secs(10)).unwrap();
        post_processor.last_check = None;
        assert!(post_processor.sync(&paths), "saved again");

        assert!(post_processor.sync(&ShaderPaths::default()), "turned off");
        assert!(!post_processor.sync(&ShaderPaths::default()), "still off");
        std::fs::remove_file(&path).unwrap();
    }
}

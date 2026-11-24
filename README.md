# egui-dither-bg

Two-pass GPU GLSL ordered-dither (`Bayer 8x8`) interactive background for `egui` / `eframe` (`glow` backend), extracted from [Covalos](https://www.matiasgs.com/work/covalos).

## Features

- **Two variants:** `DitherVariant::Field` (4-octave drifting value-noise clouds) and `DitherVariant::Bloom` (soft breathing form).
- **Interactive pointer brush:** continuous 16-sample cursor trail that lights passing cells in the accent color and fades smoothly.
- **Click ripples:** up to 6 simultaneous expanding dithered rings on pointer clicks.
- **Two-pass GPU pipeline:** Pass 1 evaluates noise, trail, and ripples once per dither cell into a small offscreen `RGBA8` texture. Pass 2 composites the `Bayer 8x8` threshold, square pixel mask, and soft vignette at full resolution. Per-frame CPU work is just uniform uploads.
- **Safe fallback check:** `egui_dither_bg::available()` flips to `false` on unsupported GL contexts (GLSL 1.20 / ES 2) so apps can fall back cleanly.

## Run the interactive demo

```bash
cargo run --example demo --release
```

Move the pointer across the window to paint the trail and click anywhere to emit ripples.

## Minimal usage

```rust
use eframe::egui;
use egui_dither_bg::{DitherFrame, DitherInput, DitherVariant};

// Store DitherInput in your eframe::App state.
let now = ctx.input(|i| i.time);
input.record(
    now,
    ctx.input(|i| i.pointer.hover_pos()),
    ctx.input(|i| i.pointer.primary_clicked()),
);

let mut frame = DitherFrame {
    variant: DitherVariant::Field,
    time: now as f32,
    ..Default::default()
};
input.fill(&mut frame, rect, now);
egui_dither_bg::paint(ui.painter(), rect, frame);
```

## Requirements

- Rust Edition 2024
- `eframe` / `egui_glow` (`0.33.3`) with OpenGL 3.3+ or OpenGL ES 3.0

## License

MIT

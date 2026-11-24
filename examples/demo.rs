use eframe::egui;
use egui_dither_bg::{DitherFrame, DitherInput, DitherVariant};

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1120.0, 680.0])
            .with_title("egui-dither-bg demo"),
        ..Default::default()
    };
    eframe::run_native(
        "egui-dither-bg",
        options,
        Box::new(|_| Ok(Box::new(DemoApp::default()))),
    )
}

struct DemoApp {
    input: DitherInput,
    variant: DitherVariant,
    pixel: f32,
    density: f32,
    strength: f32,
    light: bool,
    accent: egui::Color32,
}

impl Default for DemoApp {
    fn default() -> Self {
        Self {
            input: DitherInput::default(),
            variant: DitherVariant::Field,
            pixel: 4.0,
            density: 0.55,
            strength: 0.95,
            light: false,
            accent: egui::Color32::from_rgb(106, 176, 255),
        }
    }
}

impl eframe::App for DemoApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let now = ctx.input(|i| i.time);
        let pointer = ctx.input(|i| i.pointer.hover_pos());
        let clicked = ctx.input(|i| i.pointer.primary_clicked());
        self.input.record(now, pointer, clicked);

        let bg = if self.light {
            egui::Color32::from_rgb(243, 241, 236)
        } else {
            egui::Color32::from_rgb(11, 11, 10)
        };
        let ink = if self.light {
            egui::Rgba::from_rgb(0.16, 0.15, 0.14)
        } else {
            egui::Rgba::from_rgb(0.62, 0.65, 0.72)
        };

        egui::CentralPanel::default()
            .frame(egui::Frame::NONE.fill(bg))
            .show(ctx, |ui| {
                let rect = ui.max_rect();
                let mut shader_frame = DitherFrame {
                    variant: self.variant,
                    time: now as f32,
                    ink,
                    accent: egui::Rgba::from(self.accent),
                    strength: self.strength,
                    density: self.density,
                    pixel: self.pixel,
                    light: self.light,
                    ..Default::default()
                };
                self.input.fill(&mut shader_frame, rect, now);
                egui_dither_bg::paint(ui.painter(), rect, shader_frame);
            });

        egui::Window::new("Dither Controls")
            .default_pos([24.0, 24.0])
            .resizable(false)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.variant, DitherVariant::Field, "Field");
                    ui.selectable_value(&mut self.variant, DitherVariant::Bloom, "Bloom");
                });
                ui.add(egui::Slider::new(&mut self.pixel, 2.0..=10.0).text("Pixel"));
                ui.add(egui::Slider::new(&mut self.density, 0.1..=1.2).text("Density"));
                ui.add(egui::Slider::new(&mut self.strength, 0.2..=1.2).text("Strength"));
                ui.checkbox(&mut self.light, "Light surface");
                ui.horizontal(|ui| {
                    ui.label("Accent");
                    egui::color_picker::color_edit_button_srgba(
                        ui,
                        &mut self.accent,
                        egui::color_picker::Alpha::Opaque,
                    );
                });
                ui.label("Move cursor to paint trail. Click to emit ripples.");
            });

        ctx.request_repaint();
    }
}

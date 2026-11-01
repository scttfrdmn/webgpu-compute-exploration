use wasm_bindgen::prelude::*;
use web_sys::{HtmlCanvasElement, CanvasRenderingContext2d, ImageData};

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

#[wasm_bindgen]
pub struct MandelbrotConfig {
    width: u32,
    height: u32,
    max_iterations: u32,
    zoom: f64,
    center_x: f64,
    center_y: f64,
}

#[wasm_bindgen]
impl MandelbrotConfig {
    #[wasm_bindgen(constructor)]
    pub fn new(width: u32, height: u32) -> MandelbrotConfig {
        MandelbrotConfig {
            width,
            height,
            max_iterations: 256,
            zoom: 1.0,
            center_x: -0.5,
            center_y: 0.0,
        }
    }

    pub fn set_zoom(&mut self, zoom: f64) {
        self.zoom = zoom;
    }

    pub fn set_center(&mut self, x: f64, y: f64) {
        self.center_x = x;
        self.center_y = y;
    }

    // Used by the Measure tab (js/teach/experiments/wasm-mandelbrot.js) to time other iteration limits
    pub fn set_max_iterations(&mut self, max_iterations: u32) {
        self.max_iterations = max_iterations;
    }

    pub fn get_width(&self) -> u32 {
        self.width
    }

    pub fn get_height(&self) -> u32 {
        self.height
    }

    pub fn get_max_iterations(&self) -> u32 {
        self.max_iterations
    }

    pub fn get_shader_params(&self) -> Vec<f32> {
        vec![
            self.width as f32,
            self.height as f32,
            self.max_iterations as f32,
            self.zoom as f32,
            self.center_x as f32,
            self.center_y as f32,
        ]
    }
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (u8, u8, u8) {
    let c = v * s;
    let h_prime = (h % 360.0) / 60.0;
    let x = c * (1.0 - ((h_prime % 2.0) - 1.0).abs());
    let m = v - c;

    let (r, g, b) = match h_prime as i32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };

    (
        ((r + m) * 255.0) as u8,
        ((g + m) * 255.0) as u8,
        ((b + m) * 255.0) as u8,
    )
}

#[wasm_bindgen]
pub fn render_to_canvas(canvas: &HtmlCanvasElement, data: &[f32], config: &MandelbrotConfig) -> Result<(), JsValue> {
    let ctx = canvas
        .get_context("2d")?
        .unwrap()
        .dyn_into::<CanvasRenderingContext2d>()?;

    let width = config.width;
    let height = config.height;
    let max_iter = config.max_iterations as f32;

    let mut pixel_data = vec![0u8; (width * height * 4) as usize];

    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) as usize;
            let iterations = data[idx];

            let pixel_idx = idx * 4;

            if iterations >= max_iter {
                pixel_data[pixel_idx] = 0;
                pixel_data[pixel_idx + 1] = 0;
                pixel_data[pixel_idx + 2] = 0;
            } else {
                let hue = (iterations / max_iter * 360.0) % 360.0;
                let (r, g, b) = hsv_to_rgb(hue, 1.0, 1.0);
                pixel_data[pixel_idx] = r;
                pixel_data[pixel_idx + 1] = g;
                pixel_data[pixel_idx + 2] = b;
            }
            pixel_data[pixel_idx + 3] = 255;
        }
    }

    let image_data = ImageData::new_with_u8_clamped_array_and_sh(
        wasm_bindgen::Clamped(&pixel_data),
        width,
        height,
    )?;

    ctx.put_image_data(&image_data, 0.0, 0.0)?;

    Ok(())
}

#[wasm_bindgen]
pub fn compute_mandelbrot_cpu(config: &MandelbrotConfig) -> Vec<f32> {
    let width = config.width;
    let height = config.height;
    let max_iterations = config.max_iterations;

    let mut result = vec![0.0; (width * height) as usize];

    let scale = 4.0 / (config.zoom * width.min(height) as f64);

    for y in 0..height {
        for x in 0..width {
            let cx = config.center_x + (x as f64 - width as f64 / 2.0) * scale;
            let cy = config.center_y + (y as f64 - height as f64 / 2.0) * scale;

            let mut zx = 0.0;
            let mut zy = 0.0;
            let mut iteration = 0;

            while zx * zx + zy * zy < 4.0 && iteration < max_iterations {
                let temp = zx * zx - zy * zy + cx;
                zy = 2.0 * zx * zy + cy;
                zx = temp;
                iteration += 1;
            }

            result[(y * width + x) as usize] = iteration as f32;
        }
    }

    result
}

#[wasm_bindgen(start)]
pub fn start() {
    log("WebGPU WASM module initialized");
}

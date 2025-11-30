// Ray Marching for 3D Fractals
// Uses distance field ray marching to render volumetric fractals

const RAY_MARCH_SHADER = `
struct Uniforms {
    resolution: vec2<f32>,
    time: f32,
    power: f32,
    maxSteps: u32,
    fractalType: u32,
    cameraRotX: f32,
    cameraRotY: f32,
    cameraZoom: f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

// Mandelbulb distance estimator
fn mandelbulbDE(pos: vec3<f32>, power: f32) -> f32 {
    var z = pos;
    var dr = 1.0;
    var r = 0.0;

    for (var i = 0; i < 15; i++) {
        r = length(z);
        if (r > 2.0) { break; }

        // Convert to polar coordinates
        let theta = acos(z.z / r);
        let phi = atan2(z.y, z.x);
        dr = pow(r, power - 1.0) * power * dr + 1.0;

        // Scale and rotate the point
        let zr = pow(r, power);
        let newTheta = theta * power;
        let newPhi = phi * power;

        // Convert back to cartesian coordinates
        z = zr * vec3<f32>(
            sin(newTheta) * cos(newPhi),
            sin(newPhi) * sin(newTheta),
            cos(newTheta)
        );
        z += pos;
    }

    return 0.5 * log(r) * r / dr;
}

// Julia set distance estimator
fn juliaDE(pos: vec3<f32>) -> f32 {
    let c = vec4<f32>(0.18, 0.88, 0.24, 0.16);
    var z = vec4<f32>(pos, 0.0);
    var md2 = 1.0;
    var mz2 = dot(z, z);

    for (var i = 0; i < 12; i++) {
        md2 *= 4.0 * mz2;

        // z = z^2 + c
        z = vec4<f32>(
            z.x * z.x - dot(z.yzw, z.yzw),
            2.0 * z.x * z.yzw
        ) + c;

        mz2 = dot(z, z);
        if (mz2 > 4.0) { break; }
    }

    return 0.25 * sqrt(mz2 / md2) * log(mz2);
}

// Menger sponge distance estimator
fn mengerDE(pos: vec3<f32>) -> f32 {
    var p = pos;
    var d = max(max(abs(p.x), abs(p.y)), abs(p.z)) - 1.0;
    var s = 1.0;

    for (var i = 0; i < 4; i++) {
        let a = (abs(p) * 3.0 - 1.0) % 2.0 - 1.0;
        s *= 3.0;
        let r = abs(1.0 - 3.0 * abs(a));

        let da = max(r.x, r.y);
        let db = max(r.y, r.z);
        let dc = max(r.z, r.x);
        let c = (min(da, min(db, dc)) - 1.0) / s;

        d = max(d, c);
        p = a;
    }

    return d;
}

// Mandelbox distance estimator
fn mandelboxDE(pos: vec3<f32>) -> f32 {
    var p = pos;
    var dr = 1.0;
    let scale = 2.0;
    let fixedRadius = 1.0;
    let minRadius = 0.5;
    let fixedRadius2 = fixedRadius * fixedRadius;
    let minRadius2 = minRadius * minRadius;

    for (var i = 0; i < 12; i++) {
        // Box fold
        p = clamp(p, vec3<f32>(-1.0), vec3<f32>(1.0)) * 2.0 - p;

        // Sphere fold
        let r2 = dot(p, p);
        if (r2 < minRadius2) {
            p *= fixedRadius2 / minRadius2;
            dr *= fixedRadius2 / minRadius2;
        } else if (r2 < fixedRadius2) {
            p *= fixedRadius2 / r2;
            dr *= fixedRadius2 / r2;
        }

        // Scale and translate
        p = p * scale + pos;
        dr = dr * abs(scale) + 1.0;
    }

    return length(p) / abs(dr);
}

fn sceneSDF(pos: vec3<f32>) -> f32 {
    switch uniforms.fractalType {
        case 0u: { return mandelbulbDE(pos, uniforms.power); }
        case 1u: { return juliaDE(pos); }
        case 2u: { return mengerDE(pos); }
        case 3u: { return mandelboxDE(pos); }
        default: { return mandelbulbDE(pos, uniforms.power); }
    }
}

fn estimateNormal(p: vec3<f32>) -> vec3<f32> {
    let e = vec2<f32>(0.001, 0.0);
    return normalize(vec3<f32>(
        sceneSDF(p + e.xyy) - sceneSDF(p - e.xyy),
        sceneSDF(p + e.yxy) - sceneSDF(p - e.yxy),
        sceneSDF(p + e.yyx) - sceneSDF(p - e.yyx)
    ));
}

fn rayMarch(ro: vec3<f32>, rd: vec3<f32>) -> f32 {
    var t = 0.0;
    for (var i = 0u; i < uniforms.maxSteps; i++) {
        let pos = ro + rd * t;
        let d = sceneSDF(pos);
        if (d < 0.001) {
            return t;
        }
        if (t > 50.0) {
            break;
        }
        t += d * 0.5; // March with damping for safety
    }
    return -1.0;
}

fn rotateX(p: vec3<f32>, angle: f32) -> vec3<f32> {
    let c = cos(angle);
    let s = sin(angle);
    return vec3<f32>(p.x, c * p.y - s * p.z, s * p.y + c * p.z);
}

fn rotateY(p: vec3<f32>, angle: f32) -> vec3<f32> {
    let c = cos(angle);
    let s = sin(angle);
    return vec3<f32>(c * p.x + s * p.z, p.y, -s * p.x + c * p.z);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let coords = vec2<f32>(global_id.xy);
    if (coords.x >= uniforms.resolution.x || coords.y >= uniforms.resolution.y) {
        return;
    }

    // Normalized coordinates
    let uv = (coords - uniforms.resolution * 0.5) / uniforms.resolution.y;

    // Camera setup
    var ro = vec3<f32>(0.0, 0.0, uniforms.cameraZoom);
    ro = rotateX(ro, uniforms.cameraRotX);
    ro = rotateY(ro, uniforms.cameraRotY + uniforms.time * 0.1);

    let target = vec3<f32>(0.0, 0.0, 0.0);
    let forward = normalize(target - ro);
    let right = normalize(cross(vec3<f32>(0.0, 1.0, 0.0), forward));
    let up = cross(forward, right);

    let rd = normalize(forward + uv.x * right + uv.y * up);

    // Ray march
    let t = rayMarch(ro, rd);

    var color = vec3<f32>(0.0);
    if (t > 0.0) {
        let pos = ro + rd * t;
        let normal = estimateNormal(pos);

        // Lighting
        let lightDir = normalize(vec3<f32>(1.0, 1.0, 2.0));
        let diffuse = max(dot(normal, lightDir), 0.0);
        let specular = pow(max(dot(reflect(-lightDir, normal), -rd), 0.0), 32.0);

        // Ambient occlusion approximation
        let ao = 1.0 - f32(i32(t * 10.0)) * 0.05;

        // Color based on position and normal
        let baseColor = vec3<f32>(0.8, 0.4, 0.2) + normal * 0.2;
        color = baseColor * (diffuse * 0.7 + 0.3) * ao + specular * 0.5;

        // Fog
        color = mix(color, vec3<f32>(0.0), clamp(t / 50.0, 0.0, 1.0));
    }

    // Output to texture (would need storage texture binding)
    // For now we'll use render pass
}
`;

// We'll use a simpler full-screen render approach
const FULLSCREEN_VERTEX = `
@vertex
fn main(@builtin(vertex_index) vertexIndex: u32) -> @builtin(position) vec4<f32> {
    let pos = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(1.0, 1.0)
    );
    return vec4<f32>(pos[vertexIndex], 0.0, 1.0);
}
`;

const FRAGMENT_SHADER = RAY_MARCH_SHADER.replace('@compute @workgroup_size(8, 8)\nfn main(@builtin(global_invocation_id) global_id: vec3<u32>)',
'@fragment\nfn main(@builtin(position) fragCoord: vec4<f32>) -> @location(0) vec4<f32>') + `
    return vec4<f32>(color, 1.0);
}
`.replace('let coords = vec2<f32>(global_id.xy);', 'let coords = fragCoord.xy;');

class RayMarchingRenderer {
    constructor(device, canvas) {
        this.device = device;
        this.canvas = canvas;
        this.running = false;
        this.time = 0;

        this.fractalType = 0; // 0: Mandelbulb, 1: Julia, 2: Menger, 3: Mandelbox
        this.power = 8;
        this.maxSteps = 128;
        this.cameraRotX = 0;
        this.cameraRotY = 0;
        this.cameraZoom = 3;
        this.autoRotate = true;

        this.setupRenderPipeline();
        this.setupMouseControls();
    }

    setupMouseControls() {
        let isDragging = false;
        let lastX = 0;
        let lastY = 0;

        this.canvas.addEventListener('mousedown', (e) => {
            isDragging = true;
            lastX = e.clientX;
            lastY = e.clientY;
            this.autoRotate = false;
        });

        window.addEventListener('mousemove', (e) => {
            if (!isDragging) return;

            const dx = e.clientX - lastX;
            const dy = e.clientY - lastY;

            this.cameraRotY += dx * 0.01;
            this.cameraRotX += dy * 0.01;
            this.cameraRotX = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, this.cameraRotX));

            lastX = e.clientX;
            lastY = e.clientY;
        });

        window.addEventListener('mouseup', () => {
            isDragging = false;
        });

        this.canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            this.cameraZoom *= (1 + e.deltaY * 0.001);
            this.cameraZoom = Math.max(1.5, Math.min(6, this.cameraZoom));
        });
    }

    setupRenderPipeline() {
        this.context = this.canvas.getContext('webgpu');
        this.context.configure({
            device: this.device,
            format: 'bgra8unorm'
        });

        const shaderModule = this.device.createShaderModule({
            code: FULLSCREEN_VERTEX + '\n' + FRAGMENT_SHADER
        });

        this.uniformBuffer = this.device.createBuffer({
            size: 64,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        const bindGroupLayout = this.device.createBindGroupLayout({
            entries: [{
                binding: 0,
                visibility: GPUShaderStage.FRAGMENT,
                buffer: { type: 'uniform' }
            }]
        });

        this.bindGroup = this.device.createBindGroup({
            layout: bindGroupLayout,
            entries: [{
                binding: 0,
                resource: { buffer: this.uniformBuffer }
            }]
        });

        this.pipeline = this.device.createRenderPipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
            vertex: {
                module: shaderModule,
                entryPoint: 'main'
            },
            fragment: {
                module: shaderModule,
                entryPoint: 'main',
                targets: [{ format: 'bgra8unorm' }]
            }
        });
    }

    updateUniforms() {
        const uniforms = new Float32Array(16);
        uniforms[0] = this.canvas.width;
        uniforms[1] = this.canvas.height;
        uniforms[2] = this.time;
        uniforms[3] = this.power;
        uniforms[4] = this.maxSteps;
        uniforms[5] = this.fractalType;
        uniforms[6] = this.cameraRotX;
        uniforms[7] = this.cameraRotY;
        uniforms[8] = this.cameraZoom;

        this.device.queue.writeBuffer(this.uniformBuffer, 0, uniforms);
    }

    render() {
        if (this.autoRotate) {
            this.time += 0.016;
        }

        this.updateUniforms();

        const commandEncoder = this.device.createCommandEncoder();
        const renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: this.context.getCurrentTexture().createView(),
                loadOp: 'clear',
                storeOp: 'store',
                clearValue: { r: 0, g: 0, b: 0, a: 1 }
            }]
        });

        renderPass.setPipeline(this.pipeline);
        renderPass.setBindGroup(0, this.bindGroup);
        renderPass.draw(6);
        renderPass.end();

        this.device.queue.submit([commandEncoder.finish()]);
    }

    start() {
        this.running = true;
        this.animate();
    }

    animate() {
        if (!this.running) return;

        this.render();
        requestAnimationFrame(() => this.animate());
    }

    destroy() {
        this.running = false;
        this.uniformBuffer.destroy();
    }
}

export async function initRayMarching(gpuDevice) {
    const { device } = gpuDevice;
    const canvas = document.getElementById('canvas-fractal');
    const output = document.getElementById('output-fractal');

    const renderer = new RayMarchingRenderer(device, canvas);
    renderer.start();

    const fractalTypes = {
        'mandelbulb': 0,
        'julia': 1,
        'menger': 2,
        'mandelbox': 3
    };

    document.getElementById('fractal-type').addEventListener('change', (e) => {
        renderer.fractalType = fractalTypes[e.target.value];
        updateOutput();
    });

    const powerSlider = document.getElementById('fractal-power');
    const powerValue = document.getElementById('fractal-power-value');
    powerSlider.addEventListener('input', (e) => {
        renderer.power = parseInt(e.target.value);
        powerValue.textContent = e.target.value;
    });

    const qualitySlider = document.getElementById('fractal-quality');
    const qualityValue = document.getElementById('fractal-quality-value');
    qualitySlider.addEventListener('input', (e) => {
        renderer.maxSteps = parseInt(e.target.value);
        qualityValue.textContent = e.target.value;
        updateOutput();
    });

    function updateOutput() {
        const fractalNames = ['Mandelbulb', 'Julia Set', 'Menger Sponge', 'Mandelbox'];

        output.innerHTML = `<span class="success">✓ Ray Marching Active</span>

<span class="info">Configuration:</span>
• Fractal: ${fractalNames[renderer.fractalType]}
• Power: ${renderer.power}
• Max steps: ${renderer.maxSteps}
• Resolution: ${canvas.width} × ${canvas.height}

<span class="info">Technique:</span>
• Algorithm: Sphere tracing / Ray marching
• Distance field: Analytic distance estimators
• Lighting: Phong model with ambient occlusion
• Normals: Finite difference approximation

<span class="info">Performance:</span>
• Each pixel traces a ray through the fractal
• Cost per frame: ${canvas.width * canvas.height * renderer.maxSteps} evaluations
• Lower quality = faster rendering
• Fragment shader parallelism across all pixels

<span class="info">Fractals:</span>
• <strong>Mandelbulb</strong>: 3D extension of Mandelbrot set
• <strong>Julia Set</strong>: Quaternion Julia fractal
• <strong>Menger Sponge</strong>: Recursive cube subdivision
• <strong>Mandelbox</strong>: Box and sphere folding fractal`;
    }

    updateOutput();
    return renderer;
}

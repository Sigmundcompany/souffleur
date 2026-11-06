/* Souffleur : filtres beauté. Calcul sur la carte graphique de l'appareil, rien n'est envoyé. */
(() => {
const LOOKS = {
  none:   { name: 'Aucun',   sat: 1,    contrast: 1,    warm: 0,    fade: 0,   lift: 0 },
  nature: { name: 'Naturel', sat: 1.05, contrast: 1.03, warm: .15,  fade: 0,   lift: .03 },
  eclat:  { name: 'Éclat',   sat: 1.14, contrast: 1.05, warm: .25,  fade: 0,   lift: .08 },
  doux:   { name: 'Doux',    sat: .96,  contrast: .93,  warm: .22,  fade: .22, lift: .08 },
  chaud:  { name: 'Chaud',   sat: 1.06, contrast: 1.02, warm: .7,   fade: 0,   lift: .02 },
  froid:  { name: 'Froid',   sat: 1,    contrast: 1.03, warm: -.55, fade: 0,   lift: .02 },
  cinema: { name: 'Cinéma',  sat: .88,  contrast: 1.13, warm: -.1,  fade: .07, lift: 0 },
  nb:     { name: 'Noir et blanc', sat: 0, contrast: 1.12, warm: 0, fade: .03, lift: .02 },
};

const VERT = `attribute vec2 p; varying vec2 v;
void main(){ v = p * .5 + .5; gl_Position = vec4(p, 0., 1.); }`;

const FRAG = `precision highp float;
varying vec2 v;
uniform sampler2D tex;
uniform vec2 px;
uniform float scale, skinAmt, sharp, light, sat, contrast, warm, fade;

float skinMask(vec3 c) {
  float y = dot(c, vec3(.299, .587, .114));
  float cb = -.169 * c.r - .331 * c.g + .5 * c.b;
  float cr = .5 * c.r - .419 * c.g - .081 * c.b;
  float a = smoothstep(-.24, -.17, cb) * (1. - smoothstep(-.03, .03, cb));
  float b = smoothstep(.012, .05, cr) * (1. - smoothstep(.2, .27, cr));
  return a * b * smoothstep(.1, .24, y);
}

void main() {
  vec3 c = texture2D(tex, v).rgb;
  float m = skinMask(c);

  /* Lissage : flou qui respecte les contours, seulement sur la peau */
  if (skinAmt > 0. && m > .01) {
    vec3 sum = c; float wsum = 1.;
    for (int i = 1; i <= 24; i++) {
      float fi = float(i);
      float a = fi * 2.39996;
      float r = sqrt(fi / 24.) * 9. * scale;
      vec3 s = texture2D(tex, v + vec2(cos(a), sin(a)) * r * px).rgb;
      vec3 d = s - c;
      float w = exp(-dot(d, d) * 35.);
      sum += s * w; wsum += w;
    }
    c = mix(c, sum / wsum, m * skinAmt * .88);
  }

  /* Netteté : yeux, cheveux, vêtements, sans durcir la peau */
  if (sharp > 0.) {
    vec2 o = px * 1.3 * scale;
    vec3 n = texture2D(tex, v + vec2(o.x, 0.)).rgb + texture2D(tex, v - vec2(o.x, 0.)).rgb
           + texture2D(tex, v + vec2(0., o.y)).rgb + texture2D(tex, v - vec2(0., o.y)).rgb;
    vec3 detail = texture2D(tex, v).rgb - n * .25;
    c += detail * sharp * 1.4 * (1. - m);
  }

  /* Lumière : relève les ombres et le teint, garde les hautes lumières */
  c = pow(max(c, 0.), vec3(1. - light * .26));
  c += m * light * .02;

  /* Teint : un peu de rose et de couleur, pour ne pas blanchir le visage */
  c = mix(c, c * vec3(1.035, .975, .985), m * skinAmt);

  /* Look */
  c = (c - .5) * contrast + .5;
  float y = dot(c, vec3(.299, .587, .114));
  c = mix(vec3(y), c, sat);
  c += vec3(warm * .045, warm * .008, -warm * .05);
  c = c * (1. - fade) + fade * vec3(.16, .15, .15);

  gl_FragColor = vec4(clamp(c, 0., 1.), 1.);
}`;

class FX {
  constructor(canvas) {
    this.canvas = canvas;
    const gl = this.gl = canvas.getContext('webgl', { preserveDrawingBuffer: true, alpha: false, antialias: false, powerPreference: 'high-performance' });
    if (!gl) throw new Error('webgl');
    const sh = (type, src) => {
      const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s) || 'shader');
      return s;
    };
    const p = this.p = gl.createProgram();
    gl.attachShader(p, sh(gl.VERTEX_SHADER, VERT)); gl.attachShader(p, sh(gl.FRAGMENT_SHADER, FRAG));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error('link');
    gl.useProgram(p);
    gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    const a = gl.getAttribLocation(p, 'p');
    gl.enableVertexAttribArray(a); gl.vertexAttribPointer(a, 2, gl.FLOAT, false, 0, 0);
    gl.bindTexture(gl.TEXTURE_2D, gl.createTexture());
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
    this.u = n => gl.getUniformLocation(p, n);
  }
  /* w et h : taille de sortie. ref : hauteur de la vidéo d'origine, pour garder le même rendu en aperçu et à l'export. */
  render(source, w, h, o, ref = h) {
    const gl = this.gl, c = this.canvas, look = LOOKS[o.look] || LOOKS.none;
    if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
    gl.viewport(0, 0, w, h);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, source);
    const f = (n, v) => gl.uniform1f(this.u(n), v);
    gl.uniform2f(this.u('px'), 1 / w, 1 / h);
    f('scale', Math.max(.6, Math.min(w, h) / 720 * (h / ref)));
    f('skinAmt', o.skin / 100); f('sharp', o.sharp / 100);
    f('light', Math.min(1, o.light / 100 + look.lift));
    f('sat', look.sat); f('contrast', look.contrast); f('warm', look.warm); f('fade', look.fade);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }
}
const active = o => o.skin > 0 || o.sharp > 0 || o.light > 0 || (o.look && o.look !== 'none');
window.Beaute = { FX, LOOKS, active };
})();

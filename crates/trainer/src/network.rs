//! Neuronales Netz für Stellungsbewertung.
//!
//! Architektur: INPUT_SIZE → 128 → 64 → 4
//!
//!   • Aktivierungsfunktion Hidden: ReLU
//!   • Aktivierungsfunktion Output: tanh  (Ausgabe in [-1, 1] pro Spieler)
//!   • Initialisierung: He-Normalisierung für ReLU-Schichten
//!   • Optimierer: Adam (β1=0.9, β2=0.999, ε=1e-8)

use crate::features::INPUT_SIZE;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

pub const H1: usize = 128;
pub const H2: usize = 64;
pub const OUTPUT: usize = 4;

#[derive(Clone, Serialize, Deserialize)]
pub struct Layer {
    pub w: Vec<Vec<f32>>,
    pub b: Vec<f32>,
    // Adam erster Moment (mean)
    #[serde(skip)] pub mw: Vec<Vec<f32>>,
    #[serde(skip)] pub mb: Vec<f32>,
    // Adam zweiter Moment (unzentrierte Varianz)
    #[serde(skip)] pub vw: Vec<Vec<f32>>,
    #[serde(skip)] pub vb: Vec<f32>,
}

impl Layer {
    pub fn new_he(in_size: usize, out_size: usize, seed: u64) -> Self {
        let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
        let std = (2.0 / in_size as f32).sqrt();
        let normal = Normal::new(0.0f32, std).unwrap();
        let w = (0..out_size).map(|_| {
            (0..in_size).map(|_| normal.sample(&mut rng)).collect()
        }).collect();
        Self {
            w,
            b:  vec![0.0f32; out_size],
            mw: vec![vec![0.0f32; in_size]; out_size],
            mb: vec![0.0f32; out_size],
            vw: vec![vec![0.0f32; in_size]; out_size],
            vb: vec![0.0f32; out_size],
        }
    }

    pub fn forward_relu(&self, input: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let pre: Vec<f32> = self.w.iter().zip(&self.b).map(|(row, &bias)| {
            bias + row.iter().zip(input).map(|(&w, &x)| w * x).sum::<f32>()
        }).collect();
        let act: Vec<f32> = pre.iter().map(|&z| z.max(0.0)).collect();
        (pre, act)
    }

    pub fn forward_tanh(&self, input: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let pre: Vec<f32> = self.w.iter().zip(&self.b).map(|(row, &bias)| {
            bias + row.iter().zip(input).map(|(&w, &x)| w * x).sum::<f32>()
        }).collect();
        let act: Vec<f32> = pre.iter().map(|&z| z.tanh()).collect();
        (pre, act)
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Network {
    pub l1: Layer,
    pub l2: Layer,
    pub l3: Layer,
    pub lr: f32,
    /// Gespeichert für Rückwärtskompatibilität mit alten Checkpoints.
    #[serde(default = "default_momentum")]
    pub momentum: f32,
    pub steps: u64,
}

fn default_momentum() -> f32 { 0.9 }

const BETA1: f32 = 0.9;
const BETA2: f32 = 0.999;
const ADAM_EPS: f32 = 1e-8;

impl Network {
    pub fn new(lr: f32, _momentum: f32) -> Self {
        Self {
            l1: Layer::new_he(INPUT_SIZE, H1, 0xDEAD),
            l2: Layer::new_he(H1,         H2, 0xBEEF),
            l3: Layer::new_he(H2,     OUTPUT, 0xCAFE),
            lr,
            momentum: BETA1,
            steps: 0,
        }
    }

    pub fn forward(&self, input: &[f32]) -> [f32; 4] {
        let (_, a1) = self.l1.forward_relu(input);
        let (_, a2) = self.l2.forward_relu(&a1);
        let (_, a3) = self.l3.forward_tanh(&a2);
        [a3[0], a3[1], a3[2], a3[3]]
    }

    pub fn forward_full(&self, input: &[f32]) -> ForwardCache {
        let (z1, a1) = self.l1.forward_relu(input);
        let (z2, a2) = self.l2.forward_relu(&a1);
        let (z3, a3) = self.l3.forward_tanh(&a2);
        ForwardCache { input: input.to_vec(), z1, a1, z2, a2, z3, a3 }
    }

    pub fn backward_into_traces(
        &self,
        cache:  &ForwardCache,
        traces: &mut Traces,
        lambda: f32,
    ) {
        let d_tanh: Vec<f32> = cache.a3.iter().map(|&a| 1.0 - a * a).collect();

        for i in 0..OUTPUT {
            let d = d_tanh[i];
            for j in 0..H2 {
                traces.l3w[i][j] = lambda * traces.l3w[i][j] + d * cache.a2[j];
            }
            traces.l3b[i] = lambda * traces.l3b[i] + d;
        }

        let mut delta2 = vec![0.0f32; H2];
        for j in 0..H2 {
            let grad: f32 = (0..OUTPUT).map(|i| self.l3.w[i][j] * d_tanh[i]).sum();
            delta2[j] = grad * if cache.z2[j] > 0.0 { 1.0 } else { 0.0 };
        }
        for i in 0..H2 {
            for j in 0..H1 {
                traces.l2w[i][j] = lambda * traces.l2w[i][j] + delta2[i] * cache.a1[j];
            }
            traces.l2b[i] = lambda * traces.l2b[i] + delta2[i];
        }

        let mut delta1 = vec![0.0f32; H1];
        for j in 0..H1 {
            let grad: f32 = (0..H2).map(|i| self.l2.w[i][j] * delta2[i]).sum();
            delta1[j] = grad * if cache.z1[j] > 0.0 { 1.0 } else { 0.0 };
        }
        for i in 0..H1 {
            for j in 0..cache.input.len() {
                traces.l1w[i][j] = lambda * traces.l1w[i][j] + delta1[i] * cache.input[j];
            }
            traces.l1b[i] = lambda * traces.l1b[i] + delta1[i];
        }
    }

    /// Adam-Update: m = β1·m + (1-β1)·g; v = β2·v + (1-β2)·g²
    ///              ŵ += lr · (m/(1-β1^t)) / (sqrt(v/(1-β2^t)) + ε)
    pub fn apply_td_update(&mut self, traces: &Traces, td_error: &[f32; 4]) {
        let lr  = self.lr;
        let t   = (self.steps + 1) as f32;
        // Bias-Korrekturfaktoren (≈ 1.0 nach einigen hundert Schritten)
        let bc1 = 1.0 - BETA1.powf(t);
        let bc2 = 1.0 - BETA2.powf(t);

        macro_rules! adam_update {
            ($layer:expr, $tw:expr, $tb:expr, $out:expr, $inp:expr) => {
                for i in 0..$out {
                    let err = td_error[i % OUTPUT];
                    for j in 0..$inp {
                        let g = err * $tw[i][j];
                        $layer.mw[i][j] = BETA1 * $layer.mw[i][j] + (1.0 - BETA1) * g;
                        $layer.vw[i][j] = BETA2 * $layer.vw[i][j] + (1.0 - BETA2) * g * g;
                        let m_hat = $layer.mw[i][j] / bc1;
                        let v_hat = $layer.vw[i][j] / bc2;
                        $layer.w[i][j] += lr * m_hat / (v_hat.sqrt() + ADAM_EPS);
                    }
                    let gb = err * $tb[i];
                    $layer.mb[i] = BETA1 * $layer.mb[i] + (1.0 - BETA1) * gb;
                    $layer.vb[i] = BETA2 * $layer.vb[i] + (1.0 - BETA2) * gb * gb;
                    let m_hat = $layer.mb[i] / bc1;
                    let v_hat = $layer.vb[i] / bc2;
                    $layer.b[i] += lr * m_hat / (v_hat.sqrt() + ADAM_EPS);
                }
            };
        }

        adam_update!(self.l1, traces.l1w, traces.l1b, H1, INPUT_SIZE);
        adam_update!(self.l2, traces.l2w, traces.l2b, H2, H1);
        adam_update!(self.l3, traces.l3w, traces.l3b, OUTPUT, H2);

        self.steps += 1;
    }

    /// Initialisiert Adam-Momente nach dem Laden aus der DB.
    pub fn init_momentum(&mut self) {
        let init = |layer: &mut Layer, out: usize, inp: usize| {
            if layer.mw.is_empty() { layer.mw = vec![vec![0.0f32; inp]; out]; }
            if layer.mb.is_empty() { layer.mb = vec![0.0f32; out]; }
            if layer.vw.is_empty() { layer.vw = vec![vec![0.0f32; inp]; out]; }
            if layer.vb.is_empty() { layer.vb = vec![0.0f32; out]; }
        };
        init(&mut self.l1, H1, INPUT_SIZE);
        init(&mut self.l2, H2, H1);
        init(&mut self.l3, OUTPUT, H2);
    }

    pub fn param_count(&self) -> usize {
        INPUT_SIZE * H1 + H1 + H1 * H2 + H2 + H2 * OUTPUT + OUTPUT
    }
}

pub struct ForwardCache {
    pub input: Vec<f32>,
    pub z1: Vec<f32>, pub a1: Vec<f32>,
    pub z2: Vec<f32>, pub a2: Vec<f32>,
    pub z3: Vec<f32>, pub a3: Vec<f32>,
}

pub struct Traces {
    pub l1w: Vec<Vec<f32>>, pub l1b: Vec<f32>,
    pub l2w: Vec<Vec<f32>>, pub l2b: Vec<f32>,
    pub l3w: Vec<Vec<f32>>, pub l3b: Vec<f32>,
}

impl Traces {
    pub fn new() -> Self {
        Self {
            l1w: vec![vec![0.0; INPUT_SIZE]; H1], l1b: vec![0.0; H1],
            l2w: vec![vec![0.0; H1]; H2],         l2b: vec![0.0; H2],
            l3w: vec![vec![0.0; H2]; OUTPUT],      l3b: vec![0.0; OUTPUT],
        }
    }

    pub fn reset(&mut self) {
        for row in &mut self.l1w { row.fill(0.0); } self.l1b.fill(0.0);
        for row in &mut self.l2w { row.fill(0.0); } self.l2b.fill(0.0);
        for row in &mut self.l3w { row.fill(0.0); } self.l3b.fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_output_range() {
        let net = Network::new(0.001, 0.9);
        let out = net.forward(&vec![0.5f32; INPUT_SIZE]);
        for v in out { assert!(v >= -1.0 && v <= 1.0, "output {v} out of tanh range"); }
    }

    #[test]
    fn param_count_correct() {
        let net = Network::new(0.001, 0.9);
        assert_eq!(net.param_count(), INPUT_SIZE*H1 + H1 + H1*H2 + H2 + H2*OUTPUT + OUTPUT);
    }

    #[test]
    fn traces_reset() {
        let mut t = Traces::new();
        t.l1b[0] = 99.0;
        t.reset();
        assert_eq!(t.l1b[0], 0.0);
    }

    #[test]
    fn adam_update_changes_weights() {
        let mut net = Network::new(0.001, 0.9);
        net.init_momentum();
        let input = vec![0.5f32; INPUT_SIZE];
        let cache = net.forward_full(&input);
        let mut traces = Traces::new();
        net.backward_into_traces(&cache, &mut traces, 0.7);
        let w_before = net.l1.w[0][0];
        net.apply_td_update(&traces, &[0.1, -0.1, 0.05, -0.05]);
        assert_ne!(net.l1.w[0][0], w_before, "Adam must update weights");
    }
}

use {
    candle_core::{Device, Module, Result, Tensor},
    candle_nn::LayerNorm,
};

#[derive(Debug, Clone)]
pub struct LayerNormalization(LayerNorm);

impl LayerNormalization {
    /// `LayerNormalizartion` wraps the built-in `LayerNorm` type.
    pub fn new() -> Result<Self> {
        let w = Tensor::new(1f32, &Device::Cpu)?;
        let b = Tensor::new(0f32, &Device::Cpu)?;
        let layer_norm = LayerNorm::new(w, b, 1e-5);
        Ok(LayerNormalization(layer_norm))
    }

    /// Performs the layer normalization.
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.0.forward(xs)
    }
}

use {
    crate::activation::LayerActivation,
    candle_core::{Module, Result, Tensor},
    candle_nn::{Dropout, LayerNorm, Linear, ModuleT, VarBuilder, linear_b},
};

#[derive(Clone, Debug)]
pub enum FFLayer {
    Linear(Linear),
    GELU(LayerActivation),
    Dropout(Dropout),
    LayerNorm(LayerNorm),
}

impl Module for FFLayer {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            FFLayer::GELU(g) => g.forward(xs),
            FFLayer::Linear(l) => l.forward(xs),
            FFLayer::Dropout(d) => d.forward_t(xs, true),
            FFLayer::LayerNorm(l) => l.forward(xs),
        }
    }
}

#[derive(Clone, Debug)]
pub struct FeedForward {
    pub layers: Vec<FFLayer>,
}

impl FeedForward {
    pub fn new(in_dim: usize, _drop_p: f32, vb: VarBuilder) -> Result<Self> {
        let layers = vec![
            FFLayer::Linear(linear_b(
                in_dim,
                4_usize * in_dim,
                true,
                vb.pp("first_layer"),
            )?),
            FFLayer::GELU(LayerActivation::Activation(candle_nn::Activation::Gelu)),
            FFLayer::Linear(linear_b(
                4_usize * in_dim,
                in_dim,
                true,
                vb.pp("second_layer"),
            )?),
        ];
        Ok(Self { layers })
    }
}

impl Module for FeedForward {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = xs.clone();
        for layer in self.layers.iter() {
            xs = layer.forward(&xs)?;
        }
        Ok(xs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ff_works() {}
}

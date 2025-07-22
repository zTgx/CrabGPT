use {
    candle_core::{Result, Tensor},
    candle_nn::{Module, activation::Activation},
};

#[derive(Clone, Debug)]
pub enum LayerActivation {
    Activation(Activation),
}

impl LayerActivation {
    pub fn new(activation: Activation) -> Self {
        LayerActivation::Activation(activation)
    }
}

impl Module for LayerActivation {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            LayerActivation::Activation(a) => a.forward(xs),
        }
    }
}

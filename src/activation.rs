use {
    candle_core::{Result, Tensor},
    candle_nn::{Module, activation::Activation},
};

pub struct LayerActivation(Activation);
impl LayerActivation {
    pub fn new(activation: Activation) -> Self {
        Self(activation)
    }

    pub fn activation(&self) -> &Activation {
        &self.0
    }
}

impl Module for LayerActivation {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.0.forward(xs)
    }
}

use {
    candle_core::{Module, Result, Tensor},
    candle_nn::{Dropout, LayerNorm, Linear, ModuleT, VarBuilder, linear_b, activation::Activation},
};

#[derive(Clone, Debug)]
pub enum FFLayer {
    Linear(Linear),
    GELU(Activation),
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
            FFLayer::GELU(Activation::Gelu),
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
    use candle_core::DType;
    use candle_core::Device;
    use candle_nn::VarBuilder;
    use candle_nn::VarMap;

    #[test]
    fn ff_works() {
        let device = Device::Cpu;

        let w1b1 = VarMap::new();
        let vb = VarBuilder::from_varmap(&w1b1, DType::F32, &device);

        let in_dim = 768;
        let drop_p = 0.1;
        let ff_layer = FeedForward::new(in_dim, drop_p, vb).unwrap();

        let xs = Tensor::randn(0f32, 1f32, (2, 3, 768), &device).unwrap();

        let ys = ff_layer.forward(&xs).unwrap();

        assert_eq!(ys.shape().dims(), &[2, 3, 768]);
    }

    #[test]
    fn ff_2_works() {
        let device = Device::Cpu;

        let w1b1 = VarMap::new();
        let vb = VarBuilder::from_varmap(&w1b1, DType::F32, &device);

        let ff = FeedForward::new(768, 0.3, vb).unwrap();
        ff.layers.iter().for_each(|layer| {
            println!("Layer: {:#?}\n", layer);
        });
    }
}

use {
    candle_core::{Result, Tensor},
    candle_nn::{Dropout, Linear, Module, VarBuilder, linear},
};

#[derive(Debug, Clone)]
pub struct MultiHeadAttention {
    pub out_dim: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub w_query: Linear,
    pub w_key: Linear,
    pub w_value: Linear,
    pub out_proj: Linear,
    pub dropout: Dropout,
    pub context_length: usize,
}

impl MultiHeadAttention {
    pub fn new(
        in_dim: usize,
        out_dim: usize,
        context_length: usize,
        drop_p: f32,
        num_heads: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        assert!(out_dim % num_heads != 0);

        let head_dim = out_dim / num_heads;

        let w_query = linear(in_dim, out_dim, vb.push_prefix("w_query"))?;
        let w_key = linear(in_dim, out_dim, vb.push_prefix("w_key"))?;
        let w_value = linear(in_dim, out_dim, vb.push_prefix("w_value"))?;
        let out_proj = linear(out_dim, out_dim, vb.push_prefix("out_proj"))?;

        let dropout = Dropout::new(drop_p);

        Ok(Self {
            out_dim,
            num_heads,
            head_dim,
            w_query,
            w_key,
            w_value,
            out_proj,
            dropout,
            context_length,
        })
    }
}

impl Module for MultiHeadAttention {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        todo!()
    }
}

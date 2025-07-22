use {
    crate::{attention::MultiHeadAttention, forward::FeedForward},
    candle_core::{Module, ModuleT, Result, Tensor},
    candle_nn::{LayerNorm, VarBuilder},
};

pub struct TransformerBlock {
    attn: MultiHeadAttention,
    ff: FeedForward,
    ln1: LayerNorm,
    ln2: LayerNorm,
}

impl TransformerBlock {
    // Simplified new function for demonstration.
    // A real implementation would take a detailed config struct.
    pub fn new(
        in_dim: usize,
        num_heads: usize,
        context_length: usize,
        drop_p: f32,
        vb: VarBuilder,
    ) -> Result<Self> {
        let ln1 = candle_nn::layer_norm(in_dim, 1e-5, vb.pp("ln_1"))?;
        let ln2 = candle_nn::layer_norm(in_dim, 1e-5, vb.pp("ln_2"))?;
        let attn = MultiHeadAttention::new(
            in_dim,
            in_dim,
            context_length,
            drop_p,
            num_heads,
            vb.pp("attn"),
            true,
        )?;
        let ff = FeedForward::new(in_dim, drop_p, vb.pp("ff"))?;
        Ok(Self { attn, ff, ln1, ln2 })
    }
}

impl ModuleT for TransformerBlock {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        // Pre-LN architecture
        // First residual connection
        let residual = xs;
        let x = self.ln1.forward(xs)?;
        let x = self.attn.forward_t(&x, train)?;
        let x = (x + residual)?;

        // Second residual connection
        let residual = &x;
        let x = self.ln2.forward(&x)?;
        let x = self.ff.forward_t(&x, train)?;
        let x = (x + residual)?;
        Ok(x)
    }
}

pub struct DecoderBlock {
    masked_attn: MultiHeadAttention,
    cross_attn: MultiHeadAttention,
    ff: FeedForward,
    ln1: LayerNorm,
    ln2: LayerNorm,
    ln3: LayerNorm,
}

impl DecoderBlock {
    pub fn new(
        in_dim: usize,
        num_heads: usize,
        context_length: usize,
        drop_p: f32,
        vb: VarBuilder,
    ) -> Result<Self> {
        let ln1 = candle_nn::layer_norm(in_dim, 1e-5, vb.pp("ln_1"))?;
        let ln2 = candle_nn::layer_norm(in_dim, 1e-5, vb.pp("ln_2"))?;
        let ln3 = candle_nn::layer_norm(in_dim, 1e-5, vb.pp("ln_3"))?;
        let masked_attn = MultiHeadAttention::new(
            in_dim,
            in_dim,
            context_length,
            drop_p,
            num_heads,
            vb.pp("masked_attn"),
            true,
        )?;
        let cross_attn = MultiHeadAttention::new(
            in_dim,
            in_dim,
            context_length,
            drop_p,
            num_heads,
            vb.pp("cross_attn"),
            true,
        )?;
        let ff = FeedForward::new(in_dim, drop_p, vb.pp("ff"))?;
        Ok(Self {
            masked_attn,
            cross_attn,
            ff,
            ln1,
            ln2,
            ln3,
        })
    }

    pub fn forward_t(&self, xs: &Tensor, encoder_output: &Tensor, train: bool) -> Result<Tensor> {
        // 1. Masked Self-Attention
        let residual = xs;
        let x = self.ln1.forward(xs)?;
        // This uses the original forward_t with the causal mask
        let x = self.masked_attn.forward_t(&x, train)?;
        let x = (x + residual)?;

        // 2. Cross-Attention
        let residual = &x;
        let x_norm = self.ln2.forward(&x)?;
        // This uses the new forward_cross method
        let x = self
            .cross_attn
            .forward_cross(&x_norm, encoder_output, train)?;
        let x = (x + residual)?;

        // 3. Feed-Forward
        let residual = &x;
        let x = self.ln3.forward(&x)?;
        let x = self.ff.forward_t(&x, train)?;

        x + residual
    }
}

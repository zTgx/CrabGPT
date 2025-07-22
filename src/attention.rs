use {
    candle_core::{D, Device, Result, Tensor},
    candle_nn::{Dropout, Linear, ModuleT, VarBuilder, linear_b, ops::softmax},
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
        qkv_bias: bool,
    ) -> Result<Self> {
        assert!(
            out_dim % num_heads == 0,
            "out_dim must be divisible by num_heads"
        );

        let head_dim = out_dim / num_heads;

        let w_query = linear_b(in_dim, out_dim, qkv_bias, vb.push_prefix("w_query"))?;
        let w_key = linear_b(in_dim, out_dim, qkv_bias, vb.push_prefix("w_key"))?;
        let w_value = linear_b(in_dim, out_dim, qkv_bias, vb.push_prefix("w_value"))?;
        let out_proj = linear_b(out_dim, out_dim, qkv_bias, vb.push_prefix("out_proj"))?;

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

impl MultiHeadAttention {
    /// A forward pass for cross-attention.
    /// - `xs`: The tensor for queries (from the decoder).
    /// - `kv_xs`: The tensor for keys and values (from the encoder).
    pub fn forward_cross(&self, xs: &Tensor, kv_xs: &Tensor, train: bool) -> Result<Tensor> {
        let (b, num_tokens_q, _d_in) = xs.dims3()?;
        let (_b, num_tokens_kv, _d_in) = kv_xs.dims3()?;

        // Calculate Q from decoder input, K and V from encoder output.
        let queries = self.w_query.forward_t(xs, train)?;
        let keys = self.w_key.forward_t(kv_xs, train)?;
        let values = self.w_value.forward_t(kv_xs, train)?;

        // Reshape and transpose for multi-head attention.
        let queries = queries
            .reshape((b, num_tokens_q, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let keys = keys
            .reshape((b, num_tokens_kv, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let values = values
            .reshape((b, num_tokens_kv, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // Scaled dot-product attention. No causal mask is needed for cross-attention.
        let scaling_factor = 1. / (self.head_dim as f64).sqrt();
        let attn_scores =
            (queries.matmul(&keys.transpose(D::Minus2, D::Minus1)?)? * scaling_factor)?;
        let mut attn_weights = softmax(&attn_scores, D::Minus1)?;

        attn_weights = self.dropout.forward_t(&attn_weights, train)?;

        let context_vec = attn_weights.matmul(&values)?.transpose(1, 2)?;
        let context_vec = context_vec
            .reshape((b, num_tokens_q, self.out_dim))?
            .contiguous()?;

        self.out_proj.forward_t(&context_vec, train)
    }
}

impl ModuleT for MultiHeadAttention {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        // batch, seq_len, d_in
        let (b, num_tokens, d_in) = xs.shape().dims3()?;
        assert_eq!(d_in, self.out_dim);

        // Tensor shape: (b,num_tokens, d_out)
        let queries = self.w_query.forward_t(xs, train)?;
        let keys = self.w_key.forward_t(xs, train)?;
        let values = self.w_value.forward_t(xs, train)?;

        // Transposes from shape
        // (b, num_tokens, num_heads, head_dim) to
        // (b, num_heads, num_tokens, head_dim)
        let queries = queries
            .reshape((b, num_tokens, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let keys = keys
            .reshape((b, num_tokens, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let values = values
            .reshape((b, num_tokens, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // Calc Attention Scores
        let attn_scores = queries.matmul(&keys.transpose(D::Minus2, D::Minus1)?)?;

        // Apply Mask
        let mask = get_causal_mask(num_tokens, xs.device())?;

        // Use the mask to fill attention scores
        let masked = masked_fill(
            &attn_scores,
            &mask.broadcast_left((b, self.num_heads)).unwrap(),
            f32::NEG_INFINITY,
        )?;

        // Attention weights
        let scaling_factor = 1. / (self.head_dim as f64).sqrt();
        let mut attn_weights = softmax(&(masked * scaling_factor)?, D::Minus1)?;

        // Dropout
        attn_weights = self.dropout.forward(&attn_weights, train)?;

        // Tensor shape: (b, num_tokens, n_heads, head_dim)
        let context_vec = attn_weights.matmul(&values)?.transpose(1, 2)?;

        // Combines heads, where self.d_out = self.num_heads * self.head_dim
        let context_vec = context_vec
            .reshape((b, num_tokens, self.out_dim))?
            .contiguous()?;

        self.out_proj.forward_t(&context_vec, train)
    }
}

pub fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> Result<Tensor> {
    let shape = mask.shape();
    // println!("Shape: {:?}", shape);
    let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
    // println!("on_true: {}", on_true);
    let m = mask.where_cond(&on_true, on_false)?;
    // println!("m: {:?}", m);
    Ok(m)
}

pub fn get_causal_mask(size: usize, device: &Device) -> Result<Tensor> {
    let mask: Vec<_> = (0..size)
        .flat_map(|i| (0..size).map(move |j| u32::from(j > i)))
        .collect();
    Tensor::from_slice(&mask, (size, size), device)
}

#[cfg(test)]
mod tests {
    // TODO::
}

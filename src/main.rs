use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Module, ModuleT, VarBuilder, VarMap};
use transformer_rust::{
    embeddings::{InputEmbedding, InputEmbeddingConfig},
    pe::{PositionEmbeddingType, PositionEncoding, PositionEncodingConfig},
    transformer::{DecoderBlock, EncodeBlock},
};

fn main() -> Result<()> {
    // 1. initialize
    let device = Device::Cpu;
    let vocab_size = 100;
    let embedding_dim = 12;
    let max_position_embeddings = 100;
    let num_heads = 12;
    let context_length = 32;
    let drop_p = 0.1;

    // 2. VarBuilder
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    // 3. embedding config
    let embed_config = InputEmbeddingConfig {
        vocab_size,
        embedding_dim,
        max_position_embeddings,
    };
    let src_embedding = InputEmbedding::new(embed_config.clone(), vb.pp("src_embeddings"))?;
    let tgt_embedding = InputEmbedding::new(embed_config, vb.pp("tgt_embeddings"))?;

    let config = PositionEncodingConfig::new(
        max_position_embeddings,
        embedding_dim,
        PositionEmbeddingType::Absolute,
    );
    let src_pe_embedding = PositionEncoding::new(config.clone(), &device)?;
    let tgt_pe_embedding = PositionEncoding::new(config.clone(), &device)?;

    // 4. Encoder & Decoder
    let encoder = EncodeBlock::new(
        embedding_dim,
        num_heads,
        context_length,
        drop_p,
        vb.pp("encoder"),
    )?;
    let decoder = DecoderBlock::new(
        embedding_dim,
        num_heads,
        context_length,
        drop_p,
        vb.pp("decoder"),
    )?;

    // 5. batch_size=2, seq_len=32
    // [0, 1, 2, ..., 2*context_length-1] -> (2, context_length)
    let src_ids =
        Tensor::arange(0, (2 * context_length) as u32, &device)?.reshape((2, context_length))?;
    let tgt_ids =
        Tensor::arange(0, (2 * context_length) as u32, &device)?.reshape((2, context_length))?;

    // 6. encoder output
    let src_embedded = src_embedding.forward(&src_ids)?;
    let src_pe_embedded = src_pe_embedding.forward(context_length)?;
    let src_embedded = src_embedded.broadcast_add(&src_pe_embedded)?;

    let encoder_output = encoder.forward_t(&src_embedded, true)?;
    let tgt_pe_embedded = tgt_pe_embedding.forward(context_length)?;
    let encoder_output = encoder_output.broadcast_add(&tgt_pe_embedded)?;

    println!("Encoder output shape: {:?}", encoder_output.shape());

    // 7. decoder output
    let tgt_embedded = tgt_embedding.forward(&tgt_ids)?;
    let decoder_output = decoder.forward_t(&tgt_embedded, &encoder_output, true)?;

    println!("Decoder output shape: {:?}", decoder_output.shape());

    Ok(())
}

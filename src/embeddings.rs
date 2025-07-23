use {
    candle_core::{Result, Tensor},
    candle_nn::{Embedding, Module, VarBuilder, embedding},
};

#[derive(Debug, Clone)]
pub struct InputEmbeddingConfig {
    // vocabulary size
    pub vocab_size: usize,

    // The dimension of the vector after each token is embedded
    pub embedding_dim: usize,

    pub max_position_embeddings: usize,
}

#[derive(Debug, Clone)]
pub struct InputEmbedding {
    config: InputEmbeddingConfig,
    token_embedding_layer: Embedding,
    // pos_encoding_layer: PositionEncoding,
}

impl InputEmbedding {
    pub fn new(config: InputEmbeddingConfig, vb: VarBuilder) -> Result<Self> {
        let token_embedding_layer = embedding(
            config.vocab_size,
            config.embedding_dim,
            vb.pp("token_embeddings"),
        )?;
        println!(
            "Token embedding layer shape: {:#?}",
            token_embedding_layer.embeddings().shape()
        );

        // let pe_config = PositionEncodingConfig::new(
        //     config.max_position_embeddings,
        //     config.embedding_dim,
        //     crate::pe::PositionEmbeddingType::Absolute,
        // );
        // println!("PE config: {:#?}", pe_config);

        // let pos_encoding_layer = PositionEncoding::new(pe_config, vb.device())?;

        Ok(Self {
            config,
            token_embedding_layer,
            // pos_encoding_layer,
        })
    }
}

impl Module for InputEmbedding {
    // Look up the embedding matrix through the token ID
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let dmodel_sqrt = (self.config.embedding_dim as f32).sqrt();
        let t = Tensor::new(dmodel_sqrt, xs.device())?;

        // In the embedding layers, we multiply those weights by √dmodel.
        self.token_embedding_layer.forward(xs)?.broadcast_mul(&t)

        // // Get the sequence length from the input tensor shape
        // let (batch_size, seq_len) = xs.dims2()?;
        // println!("BatchSize: {}, seqLen: {}", batch_size, seq_len);

        // // Get token embeddings and positional embeddings
        // let token_embeddings = self.token_embedding_layer.forward(xs)?;
        // println!("Token embeddings shape: {:#?}", token_embeddings.shape());

        // // let pos_embeddings = self.pos_encoding_layer.forward(seq_len)?;
        // // println!("PE embeddings shape: {:#?}", pos_embeddings.shape());

        // println!(">>>>>>>>>>>>>>>>>>> InputEmbedding <<<<<<<<<<<<<<<<<<<<<<\n");

        // Positional embeddings are added to the token embedding vector to create the input embeddings for an LLM.
        // Eg. For token A
        // Token embedding: [1.0, 1.0, 1.0]
        // PE    embedding: [1.1, 1.2, 1.3]
        // Input embedding: = [1.0, 1.0, 1.0] + [1.1, 1.2, 1.3]
        // let input_embedding = token_embeddings.broadcast_add(&pos_embeddings);
        // input_embedding

        // Ok(token_embeddings)
    }
}

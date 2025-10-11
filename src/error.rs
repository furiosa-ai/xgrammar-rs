use thiserror::Error;

#[derive(Error, Debug)]
pub enum XGrammarErr {
    #[error("failed to load tokenizer: {0}")]
    TokenizerLoadFailed(std::io::Error),
    #[error("failed to parse tokenizer config: {0}")]
    TokenizerParseFailed(String),
    #[error("invalid tokenizer config: {0}")]
    InvalidTokenizerConfig(String),
    #[cfg(feature = "hf_hub")]
    #[error("failed to download from Hugging Face Hub: {0}")]
    HuggingFaceDownloadFailed(#[from] crate::huggingface_hub::ApiError),
    #[error("failed to parse JSON: {0}")]
    JsonParseFailed(#[from] serde_json::Error),
    #[error("missing field in JSON: {0}")]
    MissingJsonField(String),
    #[error("invalid vocab: {0}")]
    InvalidVocab(String),
    #[error("invalid structural tag: {0}")]
    InvalidStructuralTag(String),
}

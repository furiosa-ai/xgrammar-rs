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
    // -----------------------------------------------------------------------
    // Every variant below is converted directly from an error raised by the
    // underlying xgrammar C++ library. As a binding, we preserve upstream
    // behavior and pass the message through verbatim, without adding a
    // display prefix: upstream messages are already self-describing (e.g.
    // "EBNF lexer error at ...", "Invalid JSON error: ...").
    // -----------------------------------------------------------------------
    /// An untyped xgrammar error raised while constructing a `Grammar`.
    #[error("{0}")]
    InvalidGrammar(String),
    /// An untyped xgrammar error raised while compiling a grammar.
    #[error("{0}")]
    CompilationError(String),
    /// An untyped xgrammar error raised by a matcher operation.
    #[error("{0}")]
    MatcherError(String),
    // The five variants below correspond one-to-one to the upstream
    // `xgrammar::XGrammarError` subclasses
    // (thirdparty/xgrammar/include/xgrammar/exception.h), mapped via
    // `XGrammarError::GetType()`.
    /// The input JSON text is malformed.
    #[error("{0}")]
    InvalidJson(String),
    /// The JSON schema is invalid or unsatisfiable (defined upstream but not
    /// constructed as of xgrammar v0.2.3 — reserved for forward
    /// compatibility).
    #[error("{0}")]
    InvalidJsonSchema(String),
    /// The structural tag specification is invalid.
    #[error("{0}")]
    InvalidStructuralTag(String),
    /// The serialized data was produced by an incompatible xgrammar
    /// serialization version.
    #[error("{0}")]
    DeserializeVersion(String),
    /// The serialized data does not follow the expected format.
    #[error("{0}")]
    DeserializeFormat(String),
}

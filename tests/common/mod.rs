use std::sync::Once;

use hf_hub::Repo;
use tokenizers::Tokenizer;
use tracing::Level;
use xgrammar::{
    TOKENIZER_ALLOW_PATTERN,
    huggingface_hub::{self, Params, compile_glob_pattern},
};

static INIT: Once = Once::new();

/// Automatic initialization of the tracing subscriber for tests
#[ctor::ctor]
fn auto_init_subscriber() {
    INIT.call_once(|| {
        tracing_subscriber::fmt().with_max_level(Level::INFO).init();
    });
}

/// Load tokenizer from HuggingFace Hub
///
/// Downloads the model from HuggingFace Hub and loads the tokenizer from the downloaded path.
///
/// # Arguments
///
/// * `model_id` - The HuggingFace model ID (e.g., "openai/gpt-oss-20b")
///
/// # Returns
///
/// Returns a `Tokenizer` instance on success, or an error message string on failure.
///
/// # Example
///
/// ```
/// let tokenizer = load_tokenizer("openai/gpt-oss-20b")
///     .expect("Failed to load tokenizer");
/// ```
#[allow(dead_code)]
pub fn load_tokenizer(model_id: &str) -> Result<Tokenizer, String> {
    let allow_patterns = compile_glob_pattern(TOKENIZER_ALLOW_PATTERN)
        .map_err(|e| format!("Failed to compile glob pattern: {}", e))?;
    let download_options =
        Some(Params { allow_patterns: Some(allow_patterns), ..Default::default() });
    let path =
        huggingface_hub::snapshot_download(Repo::model(model_id.to_string()), download_options)
            .map_err(|e| format!("Failed to download tokenizer: {}", e))?;

    Tokenizer::from_file(path.join("tokenizer.json").to_str().unwrap())
        .map_err(|e| format!("Failed to load tokenizer from file: {}", e))
}

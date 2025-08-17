use tracing::Level;
use xgrammar::{FromPretrainedParameters, TokenizerInfo, VocabType};

const EXAONE_4_0_32B_PRETRAINED_ID: &str = "LGAI-EXAONE/EXAONE-4.0-32B";

// Shared test cases (tokenizer_id, expected_vocab_type, expected_add_prefix_space)
const TEST_TOKENIZER_CASES: &[(&str, VocabType, bool)] = &[
    ("luodian/llama-7b-hf", VocabType::ByteFallback, true),
    ("meta-llama/Llama-2-7b-chat-hf", VocabType::ByteFallback, true),
    ("meta-llama/Meta-Llama-3-8B-Instruct", VocabType::ByteLevel, false),
    ("meta-llama/Meta-Llama-3.1-8B-Instruct", VocabType::ByteLevel, false),
    // ("lmsys/vicuna-7b-v1.5", VocabType::ByteFallback, true), // no tokenizer.json
    ("NousResearch/Hermes-2-Theta-Llama-3-70B", VocabType::ByteLevel, false),
    ("NousResearch/Hermes-3-Llama-3.1-8B", VocabType::ByteLevel, false),
    ("google/gemma-2b-it", VocabType::ByteFallback, false),
    ("CohereForAI/aya-23-8B", VocabType::ByteLevel, false),
    ("deepseek-ai/DeepSeek-Coder-V2-Instruct", VocabType::ByteLevel, false),
    ("deepseek-ai/DeepSeek-V2-Chat-0628", VocabType::ByteLevel, false),
    ("deepseek-ai/deepseek-coder-7b-instruct-v1.5", VocabType::ByteLevel, false),
    ("microsoft/phi-2", VocabType::ByteLevel, false),
    ("microsoft/Phi-3-mini-4k-instruct", VocabType::ByteFallback, true),
    ("microsoft/Phi-3.5-mini-instruct", VocabType::ByteFallback, true),
    ("Qwen/Qwen1.5-4B-Chat", VocabType::ByteLevel, false),
    ("Qwen/Qwen2-7B-Instruct", VocabType::ByteLevel, false),
    // ("microsoft/Phi-3-small-8k-instruct", VocabType::Raw, false), // no tokenizer.json
    // ("Qwen/Qwen-7B-Chat", VocabType::Raw, false), // no tokenizer.json
    ("meta-llama/Llama-3.2-1B", VocabType::ByteLevel, false),
    ("google/gemma-2-2b-it", VocabType::ByteFallback, false),
    ("deepseek-ai/DeepSeek-V2.5", VocabType::ByteLevel, false),
    ("Qwen/Qwen2.5-1.5B", VocabType::ByteLevel, false),
    // ("internlm/internlm2_5-7b-chat", VocabType::ByteFallback, false), // no tokenizer.json
    ("mistralai/Mixtral-8x22B-Instruct-v0.1", VocabType::ByteFallback, true),
    // ("THUDM/glm-4-9b-chat", VocabType::Raw, false), // no tokenizer.json
    // ("THUDM/chatglm3-6b", VocabType::ByteFallback, true), // no tokenizer.json
    ("deepseek-ai/DeepSeek-R1", VocabType::ByteLevel, false),
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", VocabType::ByteLevel, false),
    ("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", VocabType::ByteLevel, false),
    ("LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct", VocabType::ByteLevel, false),
    ("LGAI-EXAONE/EXAONE-4.0-32B-FP8", VocabType::ByteLevel, false),
];

#[test]
fn test_from_pretrained() {
    tracing_subscriber::fmt().with_max_level(Level::DEBUG).init();

    let tok_info = TokenizerInfo::from_pretrained(EXAONE_4_0_32B_PRETRAINED_ID, None, None, None)
        .expect("Failed to load tokenizer info");
    assert_eq!(tok_info.get_vocab_type(), VocabType::ByteLevel);
    assert!(!tok_info.get_add_prefix_space());
    assert_eq!(tok_info.get_vocab_size(), 102400);
}

fn assert_vocab_type_prepend_space(
    tokenizer_id: &str,
    expected_vocab_type: VocabType,
    expected_add_prefix_space: bool,
) {
    use tokenizers::tokenizer::Tokenizer;

    let param = std::env::var("HF_TOKEN")
        .map(|token| FromPretrainedParameters { token: Some(token), ..Default::default() })
        .unwrap_or_default();

    let tokenizer =
        Tokenizer::from_pretrained(tokenizer_id, Some(param)).expect("Failed to load tokenizer");
    let metadata_hf = TokenizerInfo::detect_metadata_from_hf(&tokenizer.to_string(false).unwrap());
    assert_eq!(metadata_hf.vocab_type, expected_vocab_type);
    assert_eq!(metadata_hf.add_prefix_space, expected_add_prefix_space);
}

#[test]
fn test_detect_metadata_from_hf() {
    for &(tokenizer_id, expected_vocab_type, expected_add_prefix_space) in TEST_TOKENIZER_CASES {
        assert_vocab_type_prepend_space(
            tokenizer_id,
            expected_vocab_type,
            expected_add_prefix_space,
        );
    }
}

#[test]
fn test_tokenizers() {
    use tokenizers::tokenizer::Tokenizer;

    // Example usage of the tokenizers crate
    let tokenizer = Tokenizer::from_pretrained("LGAI-EXAONE/EXAONE-4.0-32B", None)
        .expect("Failed to load tokenizer");
    tokenizer
        .encode("Hello, world!", false)
        .expect("Failed to encode text")
        .get_tokens()
        .iter()
        .for_each(|token| println!("Token: {}", token));

    assert_eq!(tokenizer.get_vocab_size(false), 102400);
    assert_eq!(*tokenizer.get_vocab(false).values().max().unwrap(), 102399);
}

/// The purpose of this test is to prove the vocab map from backend string is the same as
/// one from API. This way depends on the implementation of huggingface tokenizer.
#[test]
fn test_vocab_map_from_backend_str() {
    use serde_json::Value;
    use tokenizers::tokenizer::Tokenizer;

    for &(tokenizer_id, _, _) in TEST_TOKENIZER_CASES {
        // Load tokenizer (same as other tests)
        let tokenizer = Tokenizer::from_pretrained(tokenizer_id, None)
            .unwrap_or_else(|e| panic!("failed to load tokenizer {tokenizer_id}: {e}"));

        // Original vocab map from the tokenizer API
        let vocab_from_api = tokenizer.get_vocab(false);

        // Get backend string (serialized json) and parse vocab field
        let backend_str = tokenizer.to_string(false).expect("serialize backend");
        let v: Value = serde_json::from_str(&backend_str).expect("parse backend json");
        let model = v.get("model").expect("missing model field in backend_str");
        let vocab_json = model.get("vocab").expect("missing vocab field in backend_str");
        let vocab_obj = vocab_json.as_object().expect("vocab is not an object");

        // Build vocab map from backend_str
        let mut vocab_from_backend: std::collections::HashMap<String, u32> =
            std::collections::HashMap::with_capacity(vocab_obj.len());
        for (token, id_v) in vocab_obj {
            let id = id_v
                .as_u64()
                .or_else(|| id_v.as_i64().map(|x| x as u64))
                .expect("vocab id must be a number");
            vocab_from_backend.insert(token.clone(), id as u32);
        }

        // Check basic invariants
        assert_eq!(
            vocab_from_backend.len(),
            vocab_from_api.len(),
            "vocab size mismatch for {tokenizer_id}"
        );

        // Compare every (token -> id) mapping
        for (token, token_id_from_api) in &vocab_from_api {
            let token_id_from_backend = vocab_from_backend.get(token).unwrap_or_else(|| {
                panic!("token '{}' missing in backend vocab for {tokenizer_id}", token)
            });
            assert_eq!(
                token_id_from_backend, token_id_from_api,
                "token '{}' id mismatch for {tokenizer_id}",
                token
            );
        }

        // Extra safety: max id consistency
        let max_token_id_from_api = vocab_from_api.values().max().cloned().unwrap();
        let max_token_id_from_backend = vocab_from_backend.values().max().cloned().unwrap();
        assert_eq!(
            max_token_id_from_api, max_token_id_from_backend,
            "max vocab id mismatch for {tokenizer_id}"
        );
    }
}

use tracing::Level;
use xgrammar::{FromPretrainedParameters, TokenizerInfo, VocabType};

const EXAONE_4_0_32B_PRETRAINED_ID: &str = "LGAI-EXAONE/EXAONE-4.0-32B";

#[test]
fn test_tokenizer_info() {
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
    let test_cases = [
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

    for (tokenizer_id, expected_vocab_type, expected_add_prefix_space) in test_cases {
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

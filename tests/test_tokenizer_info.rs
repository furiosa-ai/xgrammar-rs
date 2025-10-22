mod common;

use hf_hub::Repo;
use tokenizers::Tokenizer;
use xgrammar::{
    TOKENIZER_ALLOW_PATTERN, TokenizerInfo, VocabType,
    huggingface_hub::{self, Params, compile_glob_pattern},
};

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
    // ("Qwen/Qwen2-7B-Instruct", VocabType::ByteLevel, false), // no tokenizer.json
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

fn assert_metadata(
    tokenizer_info: &TokenizerInfo,
    expected_vocab_type: VocabType,
    expected_add_prefix_space: bool,
) {
    assert_eq!(tokenizer_info.get_vocab_type(), expected_vocab_type);
    assert_eq!(tokenizer_info.get_add_prefix_space(), expected_add_prefix_space);
}

/// Test to verify vocab type and add_prefix_space from tokenizer metadata
#[test]
#[ignore = "Ignored by default to avoid frequent HF hub downloads"]
fn test_tokenizer_info() {
    for &(tokenizer_id, vocab_type, add_prefix_space) in TEST_TOKENIZER_CASES {
        tracing::info!("Testing tokenizer: {}", tokenizer_id);

        let allow_patterns = compile_glob_pattern(TOKENIZER_ALLOW_PATTERN).unwrap();
        let download_options =
            Some(Params { allow_patterns: Some(allow_patterns), ..Default::default() });

        let path = huggingface_hub::snapshot_download(
            Repo::model(tokenizer_id.to_string()),
            download_options,
        )
        .unwrap();
        let tokenizer = Tokenizer::from_file(path.join("tokenizer.json").to_str().unwrap())
            .expect("Failed to load tokenizer from file");

        let tokenizer_info = TokenizerInfo::from_pretrained(tokenizer_id, None, None, None)
            .expect("Failed to get tokenizer info");
        assert_metadata(&tokenizer_info, vocab_type, add_prefix_space);

        assert_eq!(tokenizer.get_vocab_size(true), tokenizer_info.get_vocab_size() as usize);
    }
}

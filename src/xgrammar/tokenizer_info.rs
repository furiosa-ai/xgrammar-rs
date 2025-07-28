use std::{collections::HashMap, ffi::CString};

use cpp::{cpp, cpp_class};
use huggingface::hub::{DownloadOptions, Repo, RepoType, compile_glob_pattern, snapshot_download};
use tokenizers;
pub use tokenizers::FromPretrainedParameters;
use tokenizers::tokenizer::Tokenizer;

use crate::xgrammar::huggingface;

cpp! {{
    #include "xgrammar/tokenizer_info.h"
    #include <picojson.h>

    using namespace std;
    using namespace xgrammar;
    using namespace picojson;

    struct MetadataFromHF {
        VocabType vocab_type;
        bool add_prefix_space;
    };
}}

cpp_class!(
    pub unsafe struct TokenizerInfo as "xgrammar::TokenizerInfo"
);

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VocabType {
    Raw = 0,
    ByteFallback = 1,
    ByteLevel = 2,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetadataFromHF {
    pub vocab_type: VocabType,
    pub add_prefix_space: bool,
}

static TOKENIZER_GLOB_PATTERN: &[&str] = &["tokenizer_config.json"];

impl TokenizerInfo {
    #[allow(dead_code)]
    pub fn from_pretrained(
        tokenizer_id: &str,
        pretrained_params: Option<FromPretrainedParameters>,
        vocab_size: Option<usize>,
        _stop_token_ids: Option<Vec<i32>>,
    ) -> Result<TokenizerInfo, huggingface::HuggingfaceError> {
        // If fails, it must be a bug.
        let allow_patterns = compile_glob_pattern(TOKENIZER_GLOB_PATTERN)
            .expect("failed to compile the glob patterns for tokenizer files");
        let download_options =
            Some(DownloadOptions { allow_patterns: Some(allow_patterns), ..Default::default() });
        let repo = Repo::new(tokenizer_id.to_string(), RepoType::Model);
        let tokenizer_path = snapshot_download(repo, download_options)?;

        let tokenizer = Tokenizer::from_pretrained(tokenizer_id, pretrained_params)
            .expect("Failed to load tokenizer");
        let vocab_map: HashMap<String, u32> = tokenizer.get_vocab(false);

        let stop_token_ids: Vec<i32> = if let Some(stop_token_ids) = _stop_token_ids {
            // Check if the provided stop_token_ids are in the vocab_map
            stop_token_ids
        } else {
            let tokenizer_config_path = tokenizer_path.join("tokenizer_config.json");
            tracing::debug!("Reading tokenizer config from: {:?}", tokenizer_config_path);
            let reader = std::fs::File::open(tokenizer_config_path)
                .expect("Failed to open tokenizer config file");
            let tokenizer_config: serde_json::Map<String, serde_json::Value> =
                serde_json::from_reader(reader).expect("Failed to parse tokenizer config");
            let eos_token = tokenizer_config.get("eos_token").unwrap().as_str().unwrap().to_owned();
            tracing::debug!("eos_token: {:?}", eos_token);
            vocab_map
                .get(&eos_token)
                .map(|&id| vec![id as i32])
                .expect("EOS token not found in vocab") // TODO: return error
        };
        tracing::debug!("stop_token_ids: {:?}", stop_token_ids);

        // Some tokenizer don't have token id 0 or 1 or 2. So the max_id could be larger than the
        // number of tokens.
        let max_id = *vocab_map.values().max().expect("msg: Failed to get max vocab id") as usize;
        let tokenizer_vocab_size = std::cmp::max(vocab_map.len(), max_id + 1);
        let vocab_size: usize =
            if let Some(size) = vocab_size { size } else { tokenizer_vocab_size };

        // Ensure the vocab size is at least as large as the max id in the vocab map
        let mut encoded_vocab = vec![CString::new("").unwrap(); vocab_size];

        // Fill the encoded_vocab with tokens from the vocab_map
        for (token, idx) in vocab_map.iter() {
            assert!((*idx as usize) < vocab_size);
            encoded_vocab[*idx as usize] =
                CString::new(token.as_str()).expect("fail to convert a token to CString");
        }

        let encoded_vocab = vocab_map
            .keys()
            .map(|token| CString::new(token.as_str()).expect("failed to convert token to CString"))
            .collect::<Vec<CString>>();
        let encoded_vocab_ptr: Vec<_> = encoded_vocab.iter().map(|s| s.as_ptr()).collect();

        let backend_str =
            tokenizer.to_string(false).expect("fail to get the backend_str from tokenizer");
        let tokenizer_metadata = TokenizerInfo::detect_metadata_from_hf(&backend_str);

        let vocab_size_i32 = vocab_size as i32;
        let encoded_vocab_ptr_ptr = encoded_vocab_ptr.as_ptr();
        let vocab_type = tokenizer_metadata.vocab_type;
        let add_prefix_space = tokenizer_metadata.add_prefix_space;
        let stop_token_ids_ptr = stop_token_ids.as_ptr();
        let stop_token_ids_len = stop_token_ids.len();

        let tokenizer_info = cpp!(unsafe [
            encoded_vocab_ptr_ptr as "const char* const*",
            vocab_type as "xgrammar::VocabType",
            vocab_size_i32 as "int",
            stop_token_ids_ptr as "const int32_t*",
            stop_token_ids_len as "size_t",
            add_prefix_space as "bool"
        ] -> TokenizerInfo as "xgrammar::TokenizerInfo" {
            std::vector<std::string> encoded_vocab;
            for (int i = 0; i < vocab_size_i32; ++i) {
                encoded_vocab.push_back(std::string(encoded_vocab_ptr_ptr[i]));
            }
            std::vector<int32_t> stop_token_ids(stop_token_ids_ptr, stop_token_ids_ptr + stop_token_ids_len);

            return xgrammar::TokenizerInfo(
                encoded_vocab,
                vocab_type,
                vocab_size_i32,
                stop_token_ids,
                add_prefix_space
            );
        });

        Ok(tokenizer_info)
    }

    // // VocabType GetVocabType() const;
    pub fn get_vocab_type(&self) -> VocabType {
        cpp!(unsafe [self as "const xgrammar::TokenizerInfo*"] -> VocabType as "xgrammar::VocabType" {
            return self->GetVocabType();
        })
    }

    // bool GetAddPrefixSpace() const;
    pub fn get_add_prefix_space(&self) -> bool {
        cpp!(unsafe [self as "const xgrammar::TokenizerInfo*"] -> bool as "bool" {
            return self->GetAddPrefixSpace();
        })
    }

    // int GetVocabSize() const;
    pub fn get_vocab_size(&self) -> i32 {
        cpp!(unsafe [self as "const xgrammar::TokenizerInfo*"] -> i32 as "int" {
            return self->GetVocabSize();
        })
    }

    pub fn detect_metadata_from_hf(backend_str: &str) -> MetadataFromHF {
        let backend_str =
            CString::new(backend_str).expect("Failed to convert backend_str to CString");
        let backend_str_ptr = backend_str.as_ptr();

        cpp!(unsafe [backend_str_ptr as "const char*"] -> MetadataFromHF as "MetadataFromHF" {
            const std::string &backend_str(backend_str_ptr);
            std::string metadata_str = TokenizerInfo::DetectMetadataFromHF(backend_str);
            picojson::value v;
            std::string err = picojson::parse(v, metadata_str);
            if (!err.empty()) {
                throw std::runtime_error("Failed to parse metadata: " + err);
            }
            const picojson::object& metadata = v.get<picojson::object>();

            MetadataFromHF metadata_from_hf;
            metadata_from_hf.vocab_type = static_cast<xgrammar::VocabType>(metadata["vocab_type"].get<double>());
            metadata_from_hf.add_prefix_space = metadata["add_prefix_space"].get<bool>();
            return metadata_from_hf;
        })
    }
}

#[cfg(test)]
mod tests {
    use std::u32;

    use tokenizers::FromPretrainedParameters;
    use tracing::Level;
    use tracing_subscriber;

    use crate::xgrammar::tokenizer_info::{TokenizerInfo, VocabType};

    const EXAONE_4_0_32B_PRETRAINED_ID: &str = "LGAI-EXAONE/EXAONE-4.0-32B";

    #[test]
    fn test_tokenizer_info() {
        tracing_subscriber::fmt().with_max_level(Level::DEBUG).init();

        let tok_info =
            TokenizerInfo::from_pretrained(EXAONE_4_0_32B_PRETRAINED_ID, None, None, None)
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

        let tokenizer = Tokenizer::from_pretrained(tokenizer_id, Some(param))
            .expect("Failed to load tokenizer");
        let metadata_hf =
            TokenizerInfo::detect_metadata_from_hf(&tokenizer.to_string(false).unwrap());
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

    #[test]
    fn test_find_eos_token() {
        use tokenizers::tokenizer::Tokenizer;

        let tokenizer = Tokenizer::from_pretrained("LGAI-EXAONE/EXAONE-4.0-32B", None)
            .expect("Failed to load tokenizer");
        let vocab_map = tokenizer.get_vocab(false);

        // Example EOS token
        let eos_token = "endofturn";
        let mut eos_token_id: u32 = u32::MAX;
        for (token, &id) in vocab_map.iter() {
            if token.contains(eos_token) {
                println!("Found EOS token <{}>: {}", id, token);
            }
        }
        vocab_map.get("[|endofturn|]").expect("EOS token not found in vocab");
    }
}

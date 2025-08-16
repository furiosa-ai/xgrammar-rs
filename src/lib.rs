mod huggingface;

use std::{collections::HashMap, ffi::CString};

use cpp::{cpp, cpp_class};
use dlpark::{traits::TensorView, versioned::SafeManagedTensorVersioned as DLTensor};
use huggingface::hub::{DownloadOptions, Repo, RepoType, compile_glob_pattern, snapshot_download};
pub use tokenizers;
pub use tokenizers::FromPretrainedParameters;
use tokenizers::tokenizer::Tokenizer;

cpp! {{
    #include "xgrammar/xgrammar.h"
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
cpp_class!(
    pub unsafe struct GrammarCompiler as "xgrammar::GrammarCompiler"
);
cpp_class!(
    pub unsafe struct CompiledGrammar as "xgrammar::CompiledGrammar"
);
cpp_class!(
    pub unsafe struct Grammar as "xgrammar::Grammar"
);
cpp_class!(
    pub unsafe struct GrammarMatcher as "xgrammar::GrammarMatcher"
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

impl CompiledGrammar {
    pub fn get_grammar(&self) -> Grammar {
        cpp!(unsafe [self as "const xgrammar::CompiledGrammar*"] -> Grammar as "xgrammar::Grammar" {
            return self->GetGrammar();
        })
    }

    /// Return the tokenizer info associated with this compiled grammar.
    pub fn get_tokenizer_info(&self) -> TokenizerInfo {
        cpp!(unsafe [self as "const xgrammar::CompiledGrammar*"] -> TokenizerInfo as "xgrammar::TokenizerInfo" {
            return self->GetTokenizerInfo();
        })
    }

    /// Return the approximate memory usage of the grammar in bytes.
    pub fn memory_size_bytes(&self) -> usize {
        cpp!(unsafe [self as "const xgrammar::CompiledGrammar*"] -> usize as "size_t" {
            return self->MemorySizeBytes();
        })
    }
}

impl GrammarCompiler {
    pub fn new(tokenizer_info: &TokenizerInfo) -> Self {
        Self::with(tokenizer_info, None, None, None)
    }

    pub fn with(
        tokenizer_info: &TokenizerInfo,
        max_threads: Option<usize>,
        cache_enabled: Option<bool>,
        max_memory_bytes: Option<usize>,
    ) -> Self {
        let max_threads = max_threads.unwrap_or(1) as i32;
        let cache_enabled = cache_enabled.unwrap_or(true);
        let max_memory_bytes: i64 = max_memory_bytes.map(|v| v as i64).unwrap_or(-1);

        let grammar_compiler = cpp!(unsafe [
            tokenizer_info as "const xgrammar::TokenizerInfo*",
            max_threads as "int",
            cache_enabled as "bool",
            max_memory_bytes as "long long"
        ] -> GrammarCompiler as "xgrammar::GrammarCompiler" {
            return xgrammar::GrammarCompiler(
                *tokenizer_info,
                max_threads,
                cache_enabled,
                max_memory_bytes
            );
        });

        grammar_compiler
    }

    /// Get the compiled grammar for pure JSON.
    pub fn compile_builtin_json_grammar(&mut self) -> CompiledGrammar {
        cpp!(unsafe [self as "xgrammar::GrammarCompiler*"] -> CompiledGrammar as "xgrammar::CompiledGrammar" {
            return self->CompileBuiltinJSONGrammar();
        })
    }
}

impl GrammarMatcher {
    pub fn with(
        compiled_grammar: &CompiledGrammar,
        override_stop_tokens: Option<Vec<i32>>,
        terminate_without_stop_token: Option<bool>,
        max_rollback_tokens: Option<i32>,
    ) -> Self {
        // Keep it sync with the C++ implementation:
        // https://github.com/mlc-ai/xgrammar/blob/95bdfce011506ea95306b37d080115a2da3e369a/cpp/grammar_matcher.cc#L257
        let terminate_without_stop_token = terminate_without_stop_token.unwrap_or(false);
        let max_rollback_tokens = max_rollback_tokens.unwrap_or(0);
        let override_stop_tokens_ptr =
            override_stop_tokens.as_ref().map_or(std::ptr::null(), |v| v.as_ptr());
        let override_stop_tokens_len = override_stop_tokens.as_ref().map_or(0, |v| v.len());

        cpp!(unsafe [
            compiled_grammar as "const xgrammar::CompiledGrammar*",
            override_stop_tokens_ptr as "const int32_t*",
            override_stop_tokens_len as "size_t",
            terminate_without_stop_token as "bool",
            max_rollback_tokens as "int"
        ] -> GrammarMatcher as "xgrammar::GrammarMatcher" {
            std::optional<std::vector<int32_t>> opt_override_stop_tokens;
            if (override_stop_tokens_len > 0) {
                opt_override_stop_tokens = std::vector<int32_t>(
                    *override_stop_tokens_ptr,
                    *override_stop_tokens_ptr + override_stop_tokens_len
                );
            } else {
                opt_override_stop_tokens = std::nullopt;
            }

            return xgrammar::GrammarMatcher(
                *compiled_grammar,
                opt_override_stop_tokens,
                terminate_without_stop_token,
                max_rollback_tokens
            );
        })
    }

    /// Accept one token and update the state of the matcher.
    ///
    /// # Arguments
    /// * `token_id` - The id of the token to accept.
    /// * `debug_print` - If true, print debug information.
    ///
    /// # Returns
    /// * Whether the token is accepted.
    ///
    /// # Note
    /// Termination state.
    ///
    /// When the end of the root rule is reached, the matcher can only accept the stop token.
    /// The matcher is terminated after accepting the stop token, i.e. no AcceptToken or
    /// FindNextTokenMask operations can be performed. The termination state can be canceled
    /// using rollback().
    pub fn accept_token(&mut self, token_id: i32, debug_print: Option<bool>) -> bool {
        let debug_print = debug_print.unwrap_or(false);
        cpp!(unsafe [self as "xgrammar::GrammarMatcher*", token_id as "int32_t", debug_print as "bool"] -> bool as "bool" {
            return self->AcceptToken(token_id, debug_print);
        })
    }

    /// Accept a string and update the state of the matcher. The whole string is considered
    /// as one step in rollback. It is used to complement the functionality of `accept_token()`,
    /// and `accept_token()` should always be used to accept tokens.
    ///
    /// # Arguments
    /// * `input_str` - The string to be accepted.
    /// * `debug_print` - Whether to print information about the internal state of the matcher.
    ///
    /// # Returns
    /// * Whether the string is accepted.
    pub fn accept_string(&mut self, input_str: &str, debug_print: Option<bool>) -> bool {
        let debug_print = debug_print.unwrap_or(false);
        let input_str_cstring =
            CString::new(input_str).expect("Failed to convert input_str to CString");
        let input_str_ptr = input_str_cstring.as_ptr();

        cpp!(unsafe [self as "xgrammar::GrammarMatcher*", input_str_ptr as "const char*", debug_print as "bool"] -> bool as "bool" {
            return self->AcceptString(input_str_ptr, debug_print);
        })
    }

    /// Get the set of tokens that are acceptable for the next step and store them in a bitmask.
    ///
    /// # Arguments
    /// * `next_token_bitmask` - The bitmask to store the result. The bitmask must be pre-allocated
    ///   and with shape (GetBitmaskSize(),) and dtype int32.
    /// * `index` - The index of the bitmask to fill. If None, the first bitmask is filled.
    /// * `debug_print` - If true, print debug information.
    ///
    /// # Returns
    /// * Whether the bitmask need to be applied (not all-true).
    pub fn fill_next_token_bitmask(
        &mut self,
        next_token_bitmask: &mut DLTensor,
        index: Option<usize>,
        debug_print: Option<bool>,
    ) -> bool {
        let dl_tensor = next_token_bitmask.dl_tensor();
        let index = index.unwrap_or(0) as i32;
        let debug_print = debug_print.unwrap_or(false);

        cpp!(unsafe [self as "xgrammar::GrammarMatcher*", dl_tensor as "DLTensor*", index as "int32_t", debug_print as "bool"] -> bool as "bool" {
            return self->FillNextTokenBitmask(dl_tensor, index, debug_print);
        })
    }

    /// Find the jump-forward string for jump-forward decoding. This is the longest string that
    /// will be valid according to the current syntax.
    ///
    /// # Note
    /// This method does not change the grammar state.
    pub fn find_jump_forward_string(&self) -> String {
        // cpp!(unsafe [self as "const xgrammar::GrammarMatcher*"] -> String as "std::string" {
        //     return self->FindJumpForwardString();
        // })
        unimplemented!()
    }

    /// Rollback the matcher to a previous state.
    ///
    /// # Arguments
    /// * `num_tokens` - The number of tokens to rollback. It cannot exceed the current number of
    ///   steps, nor can it exceed the specified maximum number of rollback tokens.
    pub fn rollback(&mut self, num_tokens: Option<i32>) {
        let num_tokens = num_tokens.unwrap_or(1);
        cpp!(unsafe [self as "xgrammar::GrammarMatcher*", num_tokens as "int"] {
            self->Rollback(num_tokens);
        })
    }

    /// Check if the matcher has accepted the stop token and terminated.
    pub fn is_terminated(&self) -> bool {
        cpp!(unsafe [self as "const xgrammar::GrammarMatcher*"] -> bool as "bool" {
            return self->IsTerminated();
        })
    }

    /// Get the maximum number of rollback tokens allowed.
    pub fn get_max_rollback_tokens(&self) -> i32 {
        cpp!(unsafe [self as "const xgrammar::GrammarMatcher*"] -> i32 as "int" {
            return self->GetMaxRollbackTokens();
        })
    }

    /// const std::vector<int>& GetStopTokenIds() const;
    pub fn get_stop_token_ids(&self) -> Vec<i32> {
        unimplemented!()
    }
}

impl TokenizerInfo {
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
            tracing::trace!("Reading tokenizer config from: {:?}", tokenizer_config_path);
            let reader = std::fs::File::open(tokenizer_config_path)
                .expect("Failed to open tokenizer config file");
            let tokenizer_config: serde_json::Map<String, serde_json::Value> =
                serde_json::from_reader(reader).expect("Failed to parse tokenizer config");
            let eos_token = tokenizer_config.get("eos_token").unwrap().as_str().unwrap().to_owned();
            tracing::trace!("Found eos_token: {:?}", eos_token);
            vocab_map
                .get(&eos_token)
                .map(|&id| vec![id as i32])
                .expect("EOS token not found in vocab") // TODO: return error
        };
        tracing::trace!("stop_token_ids: {:?}", stop_token_ids);

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
    use tokenizers::FromPretrainedParameters;
    use tracing::Level;
    use tracing_subscriber;

    use crate::{GrammarCompiler, TokenizerInfo, VocabType};

    const EXAONE_4_0_32B_PRETRAINED_ID: &str = "LGAI-EXAONE/EXAONE-4.0-32B";

    #[test]
    fn test_grammar_compiler() {
        tracing_subscriber::fmt().with_max_level(Level::TRACE).init();
        let tok_info =
            TokenizerInfo::from_pretrained(EXAONE_4_0_32B_PRETRAINED_ID, None, None, None)
                .expect("Failed to load tokenizer info");
        let mut compiler = GrammarCompiler::new(&tok_info);
        let compiled_grammar = compiler.compile_builtin_json_grammar();

        assert_eq!(compiled_grammar.memory_size_bytes(), 380204);
        assert_eq!(compiled_grammar.get_tokenizer_info().get_vocab_size(), 102400);
        assert_eq!(compiled_grammar.get_tokenizer_info().get_vocab_type(), VocabType::ByteLevel);
    }

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
}

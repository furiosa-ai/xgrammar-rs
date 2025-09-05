mod error;
#[cfg(feature = "hf_hub")]
pub mod huggingface_hub;
mod utils;

use std::path::Path;
use std::{collections::HashMap, ffi::CString};

use cpp::{cpp, cpp_class};
use dlpark::{traits::TensorView, versioned::SafeManagedTensorVersioned as DLTensor};
use error::XGrammarErr;
use serde_json::Value;
pub use tokenizers;
pub use tokenizers::FromPretrainedParameters;

use crate::utils::get_json_field;

type Result<T> = std::result::Result<T, XGrammarErr>;

pub type VocabMap = std::collections::HashMap<String, u32>;

pub type TokenId = i32;

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

pub static TOKENIZER_FILE: &str = "tokenizer.json";
pub static TOKENIZER_CONFIG_FILE: &str = "tokenizer_config.json";
pub static TOKENIZER_ALLOW_PATTERN: &[&str] = &[TOKENIZER_FILE, TOKENIZER_CONFIG_FILE];

pub static TOKENIZER_MODEL_KEY: &str = "model";
pub static TOKENIZER_VOCAB_KEY: &str = "vocab";
pub static HF_CONFIG_EOS_TOKEN_ID_KEY: &str = "eos_token_id";

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

    /// Get the compiled grammar for a JSON schema string.
    ///
    /// # Arguments
    /// * `schema` - The JSON schema string to compile
    /// * `any_whitespace` - Whether to allow any whitespace (default: true)
    /// * `indent` - Optional indentation level
    /// * `separators` - Optional custom separators (object_separator, array_separator)
    /// * `strict_mode` - Whether to use strict mode (default: true)
    ///
    /// # Returns
    /// * A compiled grammar that can be used with GrammarMatcher
    pub fn compile_json_schema(
        &mut self,
        schema: &str,
        any_whitespace: Option<bool>,
        indent: Option<i32>,
        separators: Option<(String, String)>,
        strict_mode: Option<bool>,
    ) -> CompiledGrammar {
        let schema_cstring = CString::new(schema).expect("Failed to convert schema to CString");
        let schema_ptr = schema_cstring.as_ptr();
        let any_whitespace = any_whitespace.unwrap_or(true);
        let strict_mode = strict_mode.unwrap_or(true);
        let has_indent = indent.is_some();
        let indent_value = indent.unwrap_or(0);
        let has_separators = separators.is_some();
        
        let (_obj_sep_cstring, _array_sep_cstring, obj_sep_ptr, array_sep_ptr) = 
            if let Some((obj_sep, array_sep)) = separators {
                let obj_sep_cstring = CString::new(obj_sep).expect("Failed to convert object separator to CString");
                let array_sep_cstring = CString::new(array_sep).expect("Failed to convert array separator to CString");
                let obj_sep_ptr = obj_sep_cstring.as_ptr();
                let array_sep_ptr = array_sep_cstring.as_ptr();
                (Some(obj_sep_cstring), Some(array_sep_cstring), obj_sep_ptr, array_sep_ptr)
            } else {
                (None, None, std::ptr::null(), std::ptr::null())
            };

        cpp!(unsafe [
            self as "xgrammar::GrammarCompiler*",
            schema_ptr as "const char*",
            any_whitespace as "bool",
            has_indent as "bool",
            indent_value as "int",
            has_separators as "bool",
            obj_sep_ptr as "const char*",
            array_sep_ptr as "const char*",
            strict_mode as "bool"
        ] -> CompiledGrammar as "xgrammar::CompiledGrammar" {
            std::string schema_str(schema_ptr);
            std::optional<int> opt_indent = has_indent ? std::make_optional(indent_value) : std::nullopt;
            std::optional<std::pair<std::string, std::string>> opt_separators;
            
            if (has_separators) {
                opt_separators = std::make_pair(std::string(obj_sep_ptr), std::string(array_sep_ptr));
            } else {
                opt_separators = std::nullopt;
            }
            
            return self->CompileJSONSchema(schema_str, any_whitespace, opt_indent, opt_separators, strict_mode);
        })
    }

    /// Get the compiled grammar for a regex pattern.
    ///
    /// # Arguments
    /// * `regex` - The regex pattern string to compile
    ///
    /// # Returns
    /// * A compiled grammar that can be used with GrammarMatcher
    pub fn compile_regex(&mut self, regex: &str) -> CompiledGrammar {
        let regex_cstring = CString::new(regex).expect("Failed to convert regex to CString");
        let regex_ptr = regex_cstring.as_ptr();

        cpp!(unsafe [
            self as "xgrammar::GrammarCompiler*",
            regex_ptr as "const char*"
        ] -> CompiledGrammar as "xgrammar::CompiledGrammar" {
            std::string regex_str(regex_ptr);
            return self->CompileRegex(regex_str);
        })
    }

    /// Clear the internal cache of compiled grammars.
    /// This frees up memory used by cached compiled grammars.
    pub fn clear_cache(&mut self) {
        cpp!(unsafe [self as "xgrammar::GrammarCompiler*"] {
            self->ClearCache();
        })
    }

    /// Return the approximate memory usage of the compiler cache in bytes.
    ///
    /// # Returns
    /// * The current cache size in bytes
    pub fn get_cache_size_bytes(&self) -> i64 {
        cpp!(unsafe [self as "const xgrammar::GrammarCompiler*"] -> i64 as "long long" {
            return self->GetCacheSizeBytes();
        })
    }

    /// Return the cache limit in bytes. -1 means unlimited.
    ///
    /// # Returns
    /// * The cache limit in bytes, or -1 for unlimited
    pub fn cache_limit_bytes(&self) -> i64 {
        cpp!(unsafe [self as "const xgrammar::GrammarCompiler*"] -> i64 as "long long" {
            return self->CacheLimitBytes();
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
    fn get_config(path: &Path) -> self::Result<Value> {
        let config_path = path.join("config.json");
        let content =
            std::fs::read_to_string(&config_path).map_err(XGrammarErr::TokenizerLoadFailed)?;
        let config: Value = serde_json::from_str(&content)?;
        Ok(config)
    }

    pub fn from_backend_str(
        backend_str: &str,
        vocab_size: Option<usize>,
        stop_token_ids: Vec<TokenId>,
    ) -> self::Result<Self> {
        let backend_json: serde_json::Value =
            serde_json::from_str(backend_str).expect("Failed to parse backend string as JSON");
        let model = get_json_field(&backend_json, TOKENIZER_MODEL_KEY)?;
        let vocab_map = get_json_field(model, TOKENIZER_VOCAB_KEY)?;
        let vocab_map: HashMap<String, u32> =
            serde_json::from_value(vocab_map.clone()).map_err(|e| {
                XGrammarErr::TokenizerParseFailed(format!("Failed to parse vocab map: {}", e))
            })?;

        let max_id = vocab_map
            .values()
            .max()
            .ok_or(XGrammarErr::InvalidTokenizerConfig("Vocab map is empty".to_string()))?;
        let tokenizer_vocab_size = std::cmp::max(vocab_map.len(), (max_id + 1) as usize);
        let final_vocab_size = vocab_size.unwrap_or(tokenizer_vocab_size);

        let tokenizer_metadata = Self::detect_metadata_from_hf(backend_str);
        let vocab_type = tokenizer_metadata.vocab_type;
        let add_prefix_space = tokenizer_metadata.add_prefix_space;

        Self::new(vocab_map, vocab_type, final_vocab_size, stop_token_ids, add_prefix_space)
    }

    fn from_path<P>(
        path: P,
        vocab_size: Option<usize>,
        stop_token_ids: Option<Vec<TokenId>>,
    ) -> Result<Self>
    where
        P: AsRef<Path>,
    {
        let path = path.as_ref();
        let tokenizer_json_path = path.join(TOKENIZER_FILE);
        let backend_str = std::fs::read_to_string(&tokenizer_json_path)
            .map_err(XGrammarErr::TokenizerLoadFailed)?;

        let hf_config = Self::get_config(path)?;
        let eos_token = get_json_field(&hf_config, HF_CONFIG_EOS_TOKEN_ID_KEY)?;

        let mut stop_token_ids = stop_token_ids.unwrap_or_default();

        match eos_token {
            Value::Number(eos_token_id) => {
                if eos_token_id.is_i64() {
                    let eos_token_id = eos_token_id.as_i64().unwrap() as i32;
                    stop_token_ids.push(eos_token_id);
                } else {
                    return Err(XGrammarErr::TokenizerParseFailed(
                        "eos_token must be an integer".to_string(),
                    ));
                }
            }
            Value::Array(eos_token_ids) => {
                for token in eos_token_ids {
                    if token.is_i64() {
                        let token_id = token.as_i64().unwrap() as i32;
                        stop_token_ids.push(token_id);
                    } else {
                        return Err(XGrammarErr::TokenizerParseFailed(
                            "eos_token array must contain integers".to_string(),
                        ));
                    }
                }
            }
            _ => {
                return Err(XGrammarErr::TokenizerParseFailed(
                    "eos_token must be a string or an array of strings".to_string(),
                ));
            }
        }

        Self::from_backend_str(&backend_str, vocab_size, stop_token_ids)
    }

    #[cfg(feature = "hf_hub")]
    pub fn from_pretrained(
        tokenizer_id: &str,
        revision: Option<String>,
        vocab_size: Option<usize>,
        stop_token_ids: Option<Vec<i32>>,
    ) -> Result<TokenizerInfo> {
        use huggingface_hub::{Params, Repo, RepoType, compile_glob_pattern, snapshot_download};

        let allow_patterns = compile_glob_pattern(TOKENIZER_ALLOW_PATTERN).map_err(|e| {
            XGrammarErr::TokenizerParseFailed(format!("Failed to compile glob patterns: {}", e))
        })?;
        let download_options =
            Some(Params { allow_patterns: Some(allow_patterns), ..Default::default() });

        let repo = Repo::with_revision(
            tokenizer_id.to_string(),
            RepoType::Model,
            revision.unwrap_or("main".to_string()),
        );
        let tokenizer_dir = snapshot_download(repo, download_options)?;
        Self::from_path(tokenizer_dir, vocab_size, stop_token_ids)
    }

    fn new(
        vocab_map: HashMap<String, u32>,
        vocab_type: VocabType,
        vocab_size: usize,
        stop_token_ids: Vec<i32>,
        add_prefix_space: bool,
    ) -> self::Result<Self> {
        // Ensure the vocab size is at least as large as the max id in the vocab map
        let mut encoded_vocab = vec![CString::new("").unwrap(); vocab_size];

        // Fill the encoded_vocab with tokens from the vocab_map
        for (token, idx) in vocab_map.iter() {
            assert!(
                (*idx as usize) < vocab_size,
                "Token ID {} exceeds vocab size {}",
                idx,
                vocab_size
            );
            encoded_vocab[*idx as usize] =
                CString::new(token.as_str()).expect("fail to convert a token to CString");
        }

        let encoded_vocab_ptr: Vec<_> = encoded_vocab.iter().map(|s| s.as_ptr()).collect();
        let encoded_vocab_ptr_ptr = encoded_vocab_ptr.as_ptr();
        let vocab_size_i32 = vocab_size as i32;
        let stop_token_ids_ptr = stop_token_ids.as_ptr();
        let stop_token_ids_len = stop_token_ids.len();

        Ok(cpp!(unsafe [
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
        }))
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

    // const std::vector<std::string>& GetDecodedVocab() const;
    pub fn get_decoded_vocab(&self) -> Vec<String> {
        cpp!(unsafe [self as "const xgrammar::TokenizerInfo*"] -> Vec<String> as "std::vector<std::string>" {
            return self->GetDecodedVocab();
        })
    }

    fn detect_metadata_from_hf(backend_str: &str) -> MetadataFromHF {
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

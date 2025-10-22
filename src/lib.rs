#![recursion_limit = "256"]

mod error;
#[cfg(feature = "hf_hub")]
pub mod huggingface_hub;

use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::path::Path;
use std::str::FromStr;

use cpp::{cpp, cpp_class};
use dlpark::{traits::TensorView, versioned::SafeManagedTensorVersioned as DLTensor};
pub use error::XGrammarErr;
use serde_json::Value;
pub use tokenizers;

type Result<T> = std::result::Result<T, XGrammarErr>;

pub type VocabMap = std::collections::HashMap<String, u32>;

pub type TokenId = i32;

cpp! {{
    #include "xgrammar/xgrammar.h"
    #include <picojson.h>
    #include <cstring>

    using namespace std;
    using namespace xgrammar;
    using namespace picojson;

    struct MetadataFromHF {
        VocabType vocab_type;
        bool add_prefix_space;
    };

    struct GrammarResult {
        bool success;
        Grammar grammar;
        char* error_message;
    };

    struct CompiledGrammarResult {
        bool success;
        CompiledGrammar compiled_grammar;
        char* error_message;
    };

    struct MatcherResult {
        bool success;
        bool value;
        char* error_message;
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

#[repr(C)]
pub struct GrammarResult {
    pub success: bool,
    pub grammar: Grammar,
    pub error_message: *mut std::os::raw::c_char,
}

#[repr(C)]
pub struct CompiledGrammarResult {
    pub success: bool,
    pub compiled_grammar: CompiledGrammar,
    pub error_message: *mut std::os::raw::c_char,
}

#[repr(C)]
pub struct MatcherResult {
    pub success: bool,
    pub value: bool,
    pub error_message: *mut std::os::raw::c_char,
}

/// Helper function to safely extract and free C++ error message.
///
/// # Safety
/// The error_message_ptr must be a valid C string pointer allocated with strdup
/// and must not be null.
unsafe fn extract_and_free_error_message(error_message_ptr: *mut std::os::raw::c_char) -> String {
    // SAFETY: The caller guarantees that error_message_ptr is a valid C string
    // allocated with strdup and is not null
    unsafe {
        let msg = CStr::from_ptr(error_message_ptr).to_string_lossy().into_owned();
        libc::free(error_message_ptr as *mut libc::c_void);
        msg
    }
}

impl From<GrammarResult> for Result<Grammar> {
    fn from(result: GrammarResult) -> Self {
        if result.success {
            Ok(result.grammar)
        } else {
            let error_msg = unsafe { extract_and_free_error_message(result.error_message) };
            Err(XGrammarErr::InvalidGrammar(error_msg))
        }
    }
}

impl From<CompiledGrammarResult> for Result<CompiledGrammar> {
    fn from(result: CompiledGrammarResult) -> Self {
        if result.success {
            Ok(result.compiled_grammar)
        } else {
            let error_msg = unsafe { extract_and_free_error_message(result.error_message) };
            Err(XGrammarErr::CompilationError(error_msg))
        }
    }
}

impl From<MatcherResult> for Result<bool> {
    fn from(result: MatcherResult) -> Self {
        if result.success {
            Ok(result.value)
        } else {
            let error_msg = unsafe { extract_and_free_error_message(result.error_message) };
            Err(XGrammarErr::MatcherError(error_msg))
        }
    }
}

impl From<MatcherResult> for Result<()> {
    fn from(result: MatcherResult) -> Self {
        if result.success {
            Ok(())
        } else {
            let error_msg = unsafe { extract_and_free_error_message(result.error_message) };
            Err(XGrammarErr::MatcherError(error_msg))
        }
    }
}

pub static HF_CONFIG_FILE: &str = "config.json";
pub static TOKENIZER_FILE: &str = "tokenizer.json";
pub static TOKENIZER_CONFIG_FILE: &str = "tokenizer_config.json";
pub static GENERATION_CONFIG_FILE: &str = "generation_config.json";
pub static TOKENIZER_ALLOW_PATTERN: &[&str] =
    &[TOKENIZER_FILE, TOKENIZER_CONFIG_FILE, GENERATION_CONFIG_FILE];

pub static TOKENIZER_MODEL_KEY: &str = "model";
pub static TOKENIZER_VOCAB_KEY: &str = "vocab";
pub static EOS_TOKEN_ID_KEY: &str = "eos_token_id";

impl TokenizerInfo {
    pub fn from_backend_str(
        backend_str: &str,
        vocab_size: Option<usize>,
        stop_token_ids: Vec<TokenId>,
    ) -> self::Result<Self> {
        let tokenizer = tokenizers::Tokenizer::from_str(backend_str).map_err(|e| {
            XGrammarErr::TokenizerParseFailed(format!("failed to parse tokenizer: {}", e))
        })?;
        let vocab_map = tokenizer.get_vocab(true); // with added special tokens
        let max_id = vocab_map
            .values()
            .max()
            .ok_or(XGrammarErr::InvalidTokenizerConfig("Vocab map is empty".to_string()))?;
        let tokenizer_vocab_size = std::cmp::max(vocab_map.len(), (max_id + 1) as usize);
        if let Some(vocab_size) = vocab_size {
            if vocab_size != tokenizer_vocab_size {
                tracing::warn!(
                    "Provided vocab_size {} does not match tokenizer vocab size {}. Using provided vocab_size.",
                    vocab_size,
                    tokenizer_vocab_size
                );
            }
        }
        let final_vocab_size = vocab_size.unwrap_or(tokenizer_vocab_size);
        let tokenizer_metadata = Self::detect_metadata_from_hf(backend_str);
        let vocab_type = tokenizer_metadata.vocab_type;
        let add_prefix_space = tokenizer_metadata.add_prefix_space;

        Self::new(vocab_map, vocab_type, final_vocab_size, stop_token_ids, add_prefix_space)
    }

    pub fn parse_eos_token(path: &Path, json_key: &str) -> Option<Vec<i32>> {
        let contents = std::fs::read_to_string(path).ok()?;
        let json: Value = serde_json::from_str(&contents).ok()?;
        match json.get(json_key) {
            Some(Value::Number(num)) if num.is_i64() => Some(vec![num.as_i64().unwrap() as i32]),
            Some(Value::Array(arr)) => {
                let mut eos_tokens = Vec::new();
                for item in arr {
                    if let Value::Number(num) = item {
                        if num.is_i64() {
                            eos_tokens.push(num.as_i64().unwrap() as i32);
                        } else {
                            return None;
                        }
                    } else {
                        return None;
                    }
                }
                Some(eos_tokens)
            }
            _ => None,
        }
    }

    pub fn from_path<P>(
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

        let eos_token = Self::parse_eos_token(&path.join(GENERATION_CONFIG_FILE), EOS_TOKEN_ID_KEY)
            .or_else(|| Self::parse_eos_token(&path.join(HF_CONFIG_FILE), EOS_TOKEN_ID_KEY));

        let mut stop_token_ids = stop_token_ids.unwrap_or_default();
        stop_token_ids.extend(eos_token.unwrap_or_default());
        stop_token_ids.dedup();

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
    /// Create a new GrammarCompiler with default parameters.
    ///
    /// The GrammarCompiler is a grammar compilation utility that compiles various types of
    /// grammars into CompiledGrammar objects. It is associated with a specific tokenizer
    /// and supports caching of grammar compilation results.
    ///
    /// # Arguments
    /// * `tokenizer_info` - The tokenizer info to use for the grammar compiler
    ///
    /// # Returns
    /// * A new GrammarCompiler instance with default settings (max_threads: 1, cache enabled)
    pub fn new(tokenizer_info: &TokenizerInfo) -> Self {
        Self::with(tokenizer_info, None, None, None)
    }

    /// Create a new GrammarCompiler with custom parameters.
    ///
    /// This allows fine-grained control over compilation behavior including thread usage,
    /// caching, and memory limits.
    ///
    /// # Arguments
    /// * `tokenizer_info` - The tokenizer info to use for the grammar compiler
    /// * `max_threads` - The maximum number of threads to use for parallel compilation (default: 1)
    /// * `cache_enabled` - Whether to enable caching of compiled grammars (default: true)
    /// * `max_memory_bytes` - The maximum memory in bytes to use for caching. Use None for unlimited.
    ///
    /// # Returns
    /// * A new GrammarCompiler instance with the specified settings
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

    /// Compile a Grammar object into a CompiledGrammar.
    ///
    /// This method takes a Grammar object (which can be created from EBNF, JSON schema,
    /// regex, or structural tags) and compiles it for use with a GrammarMatcher.
    ///
    /// # Arguments
    /// * `grammar` - The grammar to compile
    ///
    /// # Returns
    /// * `Ok(CompiledGrammar)` - A compiled grammar that can be used with GrammarMatcher
    /// * `Err(XGrammarErr)` - Error if the grammar compilation fails
    ///
    /// # Errors
    /// * Returns error if the grammar is invalid or compilation fails
    ///
    /// # Example
    /// ```
    /// # use xgrammar::{Grammar, GrammarCompiler, TokenizerInfo};
    /// # fn example(tokenizer_info: &TokenizerInfo) -> xgrammar::Result<()> {
    /// let compiler = GrammarCompiler::new(tokenizer_info);
    /// let grammar = Grammar::builtin_json_grammar();
    /// let compiled = compiler.compile_grammar(&grammar)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn compile_grammar(&self, grammar: &Grammar) -> Result<CompiledGrammar> {
        let result = cpp!(unsafe [
            self as "xgrammar::GrammarCompiler*",
            grammar as "const xgrammar::Grammar*"
        ] -> CompiledGrammarResult as "CompiledGrammarResult" {
            try {
                auto compiled = self->CompileGrammar(*grammar);
                return {true, compiled, nullptr};
            } catch (const std::exception& e) {
                return {false, CompiledGrammar(NullObj()), strdup(e.what())};
            }
        });

        result.into()
    }

    /// Compile a grammar for standard JSON format.
    ///
    /// This is a convenience method that returns a compiled grammar for parsing
    /// any valid JSON without schema constraints.
    ///
    /// # Returns
    /// * `Ok(CompiledGrammar)` - A compiled grammar that matches standard JSON format
    /// * `Err(XGrammarErr)` - Error if the grammar compilation fails
    ///
    /// # Errors
    /// * Returns error if the builtin JSON grammar compilation fails (unlikely)
    ///
    /// # Example
    /// ```
    /// # use xgrammar::{GrammarCompiler, TokenizerInfo};
    /// # fn example(tokenizer_info: &TokenizerInfo) -> xgrammar::Result<()> {
    /// let compiler = GrammarCompiler::new(tokenizer_info);
    /// let compiled = compiler.compile_builtin_json_grammar()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn compile_builtin_json_grammar(&self) -> Result<CompiledGrammar> {
        let result = cpp!(unsafe [self as "xgrammar::GrammarCompiler*"] -> CompiledGrammarResult as "CompiledGrammarResult" {
            try {
                auto compiled = self->CompileBuiltinJSONGrammar();
                return {true, compiled, nullptr};
            } catch (const std::exception& e) {
                return {false, CompiledGrammar(NullObj()), strdup(e.what())};
            }
        });

        result.into()
    }

    /// Compile a grammar from a JSON schema string.
    ///
    /// This method compiles a JSON schema specification into a grammar that enforces
    /// the schema constraints during text generation.
    ///
    /// # Arguments
    /// * `schema` - The JSON schema string to compile
    /// * `any_whitespace` - Whether to allow flexible whitespace in the JSON output. None uses true
    /// * `indent` - Number of spaces for indentation. None means no indentation
    /// * `separators` - Custom separators as (object_separator, array_separator), e.g., (":", ","). None uses default separators
    /// * `strict_mode` - Whether to enforce strict JSON schema validation. None uses true
    /// * `max_whitespace_cnt` - Maximum number of consecutive whitespace characters allowed. None means no limit
    ///
    /// # Returns
    /// * `Ok(CompiledGrammar)` - A compiled grammar that can be used with GrammarMatcher
    /// * `Err(XGrammarErr)` - Error if the JSON schema is invalid or compilation fails
    ///
    /// # Errors
    /// * Returns error if the JSON schema is invalid
    /// * Returns error if the schema cannot be compiled
    ///
    /// # Example
    /// ```
    /// # use xgrammar::{GrammarCompiler, TokenizerInfo};
    /// # fn example(tokenizer_info: &TokenizerInfo) -> xgrammar::Result<()> {
    /// let compiler = GrammarCompiler::new(tokenizer_info);
    /// let schema = r#"{"type": "object", "properties": {"name": {"type": "string"}}}"#;
    /// let compiled = compiler.compile_json_schema(schema, None, None, None, None, None)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn compile_json_schema(
        &self,
        schema: &str,
        any_whitespace: Option<bool>,
        indent: Option<i32>,
        separators: Option<(String, String)>,
        strict_mode: Option<bool>,
        max_whitespace_cnt: Option<i32>,
    ) -> Result<CompiledGrammar> {
        let schema_cstring = CString::new(schema).expect("Failed to convert schema to CString");
        let schema_ptr = schema_cstring.as_ptr();
        let any_whitespace = any_whitespace.unwrap_or(true);
        let strict_mode = strict_mode.unwrap_or(true);
        let has_indent = indent.is_some();
        let indent_value = indent.unwrap_or(0);
        let has_separators = separators.is_some();
        let has_max_whitespace_cnt = max_whitespace_cnt.is_some();
        let max_whitespace_cnt_value = max_whitespace_cnt.unwrap_or(0);

        let (_obj_sep_cstring, _array_sep_cstring, obj_sep_ptr, array_sep_ptr) =
            if let Some((obj_sep, array_sep)) = separators {
                let obj_sep_cstring =
                    CString::new(obj_sep).expect("Failed to convert object separator to CString");
                let array_sep_cstring =
                    CString::new(array_sep).expect("Failed to convert array separator to CString");
                let obj_sep_ptr = obj_sep_cstring.as_ptr();
                let array_sep_ptr = array_sep_cstring.as_ptr();
                (Some(obj_sep_cstring), Some(array_sep_cstring), obj_sep_ptr, array_sep_ptr)
            } else {
                (None, None, std::ptr::null(), std::ptr::null())
            };

        let result = cpp!(unsafe [
            self as "xgrammar::GrammarCompiler*",
            schema_ptr as "const char*",
            any_whitespace as "bool",
            has_indent as "bool",
            indent_value as "int",
            has_separators as "bool",
            obj_sep_ptr as "const char*",
            array_sep_ptr as "const char*",
            strict_mode as "bool",
            has_max_whitespace_cnt as "bool",
            max_whitespace_cnt_value as "int"
        ] -> CompiledGrammarResult as "CompiledGrammarResult" {
            try {
                std::string schema_str(schema_ptr);
                std::optional<int> opt_indent = has_indent ? std::make_optional(indent_value) : std::nullopt;
                std::optional<std::pair<std::string, std::string>> opt_separators;

                if (has_separators) {
                    opt_separators = std::make_pair(std::string(obj_sep_ptr), std::string(array_sep_ptr));
                } else {
                    opt_separators = std::nullopt;
                }

                std::optional<int> opt_max_whitespace_cnt = has_max_whitespace_cnt ? std::make_optional(max_whitespace_cnt_value) : std::nullopt;

                auto compiled = self->CompileJSONSchema(schema_str, any_whitespace, opt_indent, opt_separators, strict_mode, opt_max_whitespace_cnt);
                return {true, compiled, nullptr};
            } catch (const std::exception& e) {
                return {false, CompiledGrammar(NullObj()), strdup(e.what())};
            }
        });

        result.into()
    }

    /// Compile a grammar from a regular expression pattern.
    ///
    /// This method compiles a regex pattern into a grammar that matches text
    /// conforming to the specified pattern.
    ///
    /// # Arguments
    /// * `regex` - The regex pattern string to compile
    ///
    /// # Returns
    /// * `Ok(CompiledGrammar)` - A compiled grammar that can be used with GrammarMatcher
    /// * `Err(XGrammarErr)` - Error if the regex pattern is invalid or compilation fails
    ///
    /// # Errors
    /// * Returns error if the regex pattern is invalid
    /// * Returns error if the regex cannot be compiled
    ///
    /// # Example
    /// ```
    /// # use xgrammar::{GrammarCompiler, TokenizerInfo};
    /// # fn example(tokenizer_info: &TokenizerInfo) -> xgrammar::Result<()> {
    /// let compiler = GrammarCompiler::new(tokenizer_info);
    /// let compiled = compiler.compile_regex(r"[a-z]+@[a-z]+\.[a-z]+")?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn compile_regex(&self, regex: &str) -> Result<CompiledGrammar> {
        let regex_cstring = CString::new(regex).expect("Failed to convert regex to CString");
        let regex_ptr = regex_cstring.as_ptr();

        let result = cpp!(unsafe [
            self as "xgrammar::GrammarCompiler*",
            regex_ptr as "const char*"
        ] -> CompiledGrammarResult as "CompiledGrammarResult" {
            try {
                std::string regex_str(regex_ptr);
                auto compiled = self->CompileRegex(regex_str);
                return {true, compiled, nullptr};
            } catch (const std::exception& e) {
                return {false, CompiledGrammar(NullObj()), strdup(e.what())};
            }
        });

        result.into()
    }

    /// Clear the internal cache of compiled grammars.
    /// This frees up memory used by cached compiled grammars.
    pub fn clear_cache(&self) {
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

    /// Compile a grammar from a structural tag JSON string.
    ///
    /// This method compiles a structural tag specification provided as a JSON string into
    /// a grammar that can be used with a GrammarMatcher. The structural tag allows for
    /// structured text generation with specific formatting tags and schemas.
    ///
    /// # Arguments
    /// * `structural_tag_json` - A JSON string specifying the structural tag configuration.
    ///   The JSON should contain the structural tag items and triggers.
    ///
    /// # Returns
    /// * `Ok(CompiledGrammar)` - A compiled grammar that can be used with GrammarMatcher
    /// * `Err(XGrammarErr)` - Error if the structural tag is invalid or compilation fails
    ///
    /// # Errors
    /// * Returns error if the structural tag JSON is invalid
    /// * Returns error if the structural tag cannot be compiled
    ///
    /// # Example
    /// ```no_run
    /// # use xgrammar::{GrammarCompiler, TokenizerInfo};
    /// # fn example(tokenizer_info: &TokenizerInfo) -> xgrammar::Result<()> {
    /// let compiler = GrammarCompiler::new(tokenizer_info);
    /// let structural_tag_json = r#"{"tags": [{"begin": "<start>", "schema": "{}", "end": "</start>"}], "triggers": ["trigger1"]}"#;
    /// let compiled_grammar = compiler.compile_structural_tag(structural_tag_json)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn compile_structural_tag(&self, structural_tag_json: &str) -> Result<CompiledGrammar> {
        let structural_tag_json_cstring = CString::new(structural_tag_json)
            .expect("Failed to convert structural_tag_json to CString");
        let structural_tag_json_ptr = structural_tag_json_cstring.as_ptr();

        let result = cpp!(unsafe [
            self as "xgrammar::GrammarCompiler*",
            structural_tag_json_ptr as "const char*"
        ] -> CompiledGrammarResult as "CompiledGrammarResult" {
            try {
                std::string structural_tag_json_str(structural_tag_json_ptr);
                auto compiled = self->CompileStructuralTag(structural_tag_json_str);
                return {true, compiled, nullptr};
            } catch (const std::exception& e) {
                return {false, CompiledGrammar(NullObj()), strdup(e.what())};
            }
        });

        result.into()
    }
}

/// Represents a context-free grammar for grammar-guided text generation.
///
/// The Grammar struct supports Extended Backus-Naur Form (EBNF) grammar specifications
/// following the GBNF specification from llama.cpp. It provides flexible grammar generation
/// and manipulation for constrained text generation tasks.
///
/// # Construction Methods
///
/// Grammar can be constructed from various sources:
/// - [`Grammar::from_ebnf`]: From EBNF grammar strings
/// - [`Grammar::from_json_schema`]: From JSON schema specifications
/// - [`Grammar::from_regex`]: From regular expression patterns
/// - [`Grammar::from_structural_tag`]: From structural tags with embedded schemas
/// - [`Grammar::builtin_json_grammar`]: Standard JSON grammar
///
/// # Grammar Operations
///
/// Multiple grammars can be combined using:
/// - [`Grammar::union`]: Creates a grammar matching any of the input grammars (equivalent to `|` operator)
/// - [`Grammar::concat`]: Creates a grammar matching concatenated sequences (equivalent to `+` operator)
impl Grammar {
    /// Construct a grammar from an EBNF-formatted string.
    ///
    /// This method creates a context-free grammar from an Extended Backus-Naur Form (EBNF)
    /// specification. The grammar follows the GBNF specification from llama.cpp.
    ///
    /// # Arguments
    /// * `ebnf_string` - The EBNF grammar specification string
    /// * `root_rule_name` - The name of the root rule to use as the entry point. If None, uses "root"
    ///
    /// # Returns
    /// * `Ok(Grammar)` - A Grammar object constructed from the EBNF specification
    /// * `Err(XGrammarErr)` - Error if the EBNF string is invalid or malformed
    ///
    /// # Errors
    /// * Returns error if the EBNF string contains syntax errors
    /// * Returns error if the root rule is not defined
    /// * Returns error if there are undefined rule references
    ///
    /// # Example
    /// ```
    /// # use xgrammar::Grammar;
    /// let ebnf = r#"
    /// root ::= "Hello, " name "!"
    /// name ::= [A-Z][a-z]+
    /// "#;
    /// let grammar = Grammar::from_ebnf(ebnf, Some("root")).unwrap();
    /// assert!(!grammar.is_null());
    ///
    /// // Invalid EBNF will return an error
    /// let invalid_ebnf = r#"root ::= "unterminated string"#;
    /// assert!(Grammar::from_ebnf(invalid_ebnf, Some("root")).is_err());
    /// ```
    pub fn from_ebnf(ebnf_string: &str, root_rule_name: Option<&str>) -> Result<Self> {
        let ebnf_string_cstring =
            CString::new(ebnf_string).expect("Failed to convert ebnf_string to CString");
        let ebnf_string_ptr = ebnf_string_cstring.as_ptr();
        let root_rule_name = root_rule_name.unwrap_or("root");
        let root_rule_name_cstring =
            CString::new(root_rule_name).expect("Failed to convert root_rule_name to CString");
        let root_rule_name_ptr = root_rule_name_cstring.as_ptr();

        let result = cpp!(unsafe [
            ebnf_string_ptr as "const char*",
            root_rule_name_ptr as "const char*"
        ] -> GrammarResult as "GrammarResult" {
            try {
                auto grammar = Grammar::FromEBNF(string(ebnf_string_ptr), string(root_rule_name_ptr));
                return {true, grammar, nullptr};
            } catch (const std::exception& e) {
                return {false, Grammar(NullObj()), strdup(e.what())};
            }
        });

        result.into()
    }

    /// Construct a grammar from a JSON schema string.
    ///
    /// This method creates a grammar from a JSON schema specification that enforces schema
    /// constraints during text generation. The schema can be in JSON string format or
    /// represent a Pydantic-style model structure.
    ///
    /// # Arguments
    /// * `schema` - The JSON schema string defining the structure to enforce
    /// * `any_whitespace` - Whether to allow flexible whitespace in the JSON output. When true,
    ///   any amount of whitespace is allowed between tokens
    /// * `indent` - Number of spaces for indentation in the JSON output. When specified,
    ///   produces formatted JSON with the given indentation level
    /// * `separators` - Custom separators for JSON formatting as (item_separator, key_separator).
    ///   For example, `(":", ",")` produces compact JSON. When None, uses standard JSON separators
    /// * `strict_mode` - Whether to enforce strict JSON schema validation. When true, ensures
    ///   all schema constraints are strictly enforced
    /// * `max_whitespace_cnt` - Maximum number of consecutive whitespace characters allowed.
    ///   Useful for preventing excessive whitespace in generated output
    /// * `print_converted_ebnf` - Whether to print the converted EBNF grammar for debugging purposes
    ///
    /// # Returns
    /// * `Ok(Grammar)` - A Grammar object that enforces the JSON schema constraints
    /// * `Err(XGrammarErr)` - Error if the JSON schema is invalid or malformed
    ///
    /// # Errors
    /// * Returns error if the JSON schema is invalid
    /// * Returns error if the schema cannot be converted to EBNF
    ///
    /// # Example
    /// ```
    /// # use xgrammar::Grammar;
    /// let schema = r#"{
    ///   "type": "object",
    ///   "properties": {
    ///     "name": {"type": "string"},
    ///     "age": {"type": "integer"}
    ///   },
    ///   "required": ["name", "age"]
    /// }"#;
    /// let grammar = Grammar::from_json_schema(
    ///     schema,
    ///     Some(true),    // allow flexible whitespace
    ///     Some(2),       // 2-space indentation
    ///     None,          // default separators
    ///     Some(true),    // strict mode
    ///     None,          // no whitespace limit
    ///     Some(false)    // don't print EBNF
    /// ).unwrap();
    /// assert!(!grammar.is_null());
    ///
    /// // Invalid JSON schema will return an error
    /// let invalid_schema = r#"{ invalid json }"#;
    /// assert!(Grammar::from_json_schema(invalid_schema, None, None, None, None, None, None).is_err());
    /// ```
    pub fn from_json_schema(
        schema: &str,
        any_whitespace: Option<bool>,
        indent: Option<i32>,
        separators: Option<(String, String)>,
        strict_mode: Option<bool>,
        max_whitespace_cnt: Option<i32>,
        print_converted_ebnf: Option<bool>,
    ) -> Result<Self> {
        let schema_cstring = CString::new(schema).expect("Failed to convert schema to CString");
        let schema_ptr = schema_cstring.as_ptr();
        let any_whitespace = any_whitespace.unwrap_or(true);
        let strict_mode = strict_mode.unwrap_or(true);
        let print_converted_ebnf = print_converted_ebnf.unwrap_or(false);
        let has_indent = indent.is_some();
        let indent_value = indent.unwrap_or(0);
        let has_separators = separators.is_some();
        let has_max_whitespace_cnt = max_whitespace_cnt.is_some();
        let max_whitespace_cnt_value = max_whitespace_cnt.unwrap_or(0);

        let (_obj_sep_cstring, _array_sep_cstring, obj_sep_ptr, array_sep_ptr) =
            if let Some((obj_sep, array_sep)) = separators {
                let obj_sep_cstring =
                    CString::new(obj_sep).expect("Failed to convert object separator to CString");
                let array_sep_cstring =
                    CString::new(array_sep).expect("Failed to convert array separator to CString");
                let obj_sep_ptr = obj_sep_cstring.as_ptr();
                let array_sep_ptr = array_sep_cstring.as_ptr();
                (Some(obj_sep_cstring), Some(array_sep_cstring), obj_sep_ptr, array_sep_ptr)
            } else {
                (None, None, std::ptr::null(), std::ptr::null())
            };

        let result = cpp!(unsafe [
            schema_ptr as "const char*",
            any_whitespace as "bool",
            has_indent as "bool",
            indent_value as "int",
            has_separators as "bool",
            obj_sep_ptr as "const char*",
            array_sep_ptr as "const char*",
            strict_mode as "bool",
            has_max_whitespace_cnt as "bool",
            max_whitespace_cnt_value as "int",
            print_converted_ebnf as "bool"
        ] -> GrammarResult as "GrammarResult" {
            try {
                std::string schema_str(schema_ptr);
                std::optional<int> opt_indent = has_indent ? std::make_optional(indent_value) : std::nullopt;
                std::optional<std::pair<std::string, std::string>> opt_separators;

                if (has_separators) {
                    opt_separators = std::make_pair(std::string(obj_sep_ptr), std::string(array_sep_ptr));
                } else {
                    opt_separators = std::nullopt;
                }

                std::optional<int> opt_max_whitespace_cnt = has_max_whitespace_cnt ? std::make_optional(max_whitespace_cnt_value) : std::nullopt;

                auto grammar = Grammar::FromJSONSchema(
                    schema_str,
                    any_whitespace,
                    opt_indent,
                    opt_separators,
                    strict_mode,
                    opt_max_whitespace_cnt,
                    print_converted_ebnf
                );
                return {true, grammar, nullptr};
            } catch (const std::exception& e) {
                return {false, Grammar(NullObj()), strdup(e.what())};
            }
        });

        result.into()
    }

    /// Construct a grammar from a regular expression string.
    ///
    /// This method creates a grammar by converting a regular expression pattern into
    /// an EBNF grammar specification. The resulting grammar matches text conforming
    /// to the specified regex pattern.
    ///
    /// # Arguments
    /// * `regex` - The regular expression pattern string to convert
    /// * `print_converted_ebnf` - Whether to print the converted EBNF grammar for debugging purposes
    ///
    /// # Returns
    /// * `Ok(Grammar)` - A Grammar object that matches the regex pattern
    /// * `Err(XGrammarErr)` - Error if the regex pattern is invalid or malformed
    ///
    /// # Errors
    /// * Returns error if the regex pattern is invalid
    /// * Returns error if the regex cannot be converted to EBNF
    ///
    /// # Example
    /// ```
    /// # use xgrammar::Grammar;
    /// // Match email-like patterns
    /// let grammar = Grammar::from_regex(r"[a-z]+@[a-z]+\.[a-z]+", Some(false)).unwrap();
    /// assert!(!grammar.is_null());
    ///
    /// // Invalid regex will return an error
    /// let invalid_regex = r"[";
    /// assert!(Grammar::from_regex(invalid_regex, Some(false)).is_err());
    /// ```
    pub fn from_regex(regex: &str, print_converted_ebnf: Option<bool>) -> Result<Self> {
        let regex_cstring = CString::new(regex).expect("Failed to convert regex to CString");
        let regex_ptr = regex_cstring.as_ptr();
        let print_converted_ebnf = print_converted_ebnf.unwrap_or(false);

        let result = cpp!(unsafe [
            regex_ptr as "const char*",
            print_converted_ebnf as "bool"
        ] -> GrammarResult as "GrammarResult" {
            try {
                auto grammar = Grammar::FromRegex(string(regex_ptr), print_converted_ebnf);
                return {true, grammar, nullptr};
            } catch (const std::exception& e) {
                return {false, Grammar(NullObj()), strdup(e.what())};
            }
        });

        result.into()
    }

    /// Construct a grammar from a structural tag JSON string.
    ///
    /// This method creates a grammar from structural tags that enable grammar-guided generation
    /// with specific formatting markers. Structural tags are useful for dispatching between
    /// different grammars based on trigger tokens and wrapping content with specific begin/end tags.
    ///
    /// The structural tag format supports:
    /// - Single tag specification with begin marker, JSON schema, and end marker
    /// - Multiple tags for grammar dispatching based on triggers
    /// - Legacy tag/trigger pattern support
    ///
    /// # Arguments
    /// * `structural_tag_json` - A JSON string specifying the structural tag configuration.
    ///   The JSON should contain structural tag items with `begin`, `schema`, and `end` fields,
    ///   and optionally `triggers` for grammar dispatching.
    ///
    /// # Returns
    /// * `Ok(Grammar)` if the JSON is valid and the grammar was successfully created
    /// * `Err(XGrammarErr)` if the JSON is invalid or the structural tag is malformed
    ///
    /// # Example
    /// ```
    /// # use xgrammar::Grammar;
    /// use serde_json::json;
    ///
    /// // Triggered tags example for tool calling with multiple functions
    /// let structural_tag = json!({
    ///     "format": {
    ///         "type": "triggered_tags",
    ///         "triggers": ["<function="],
    ///         "tags": [
    ///             {
    ///                 "begin": "<function=get_weather>",
    ///                 "content": {
    ///                     "type": "json_schema",
    ///                     "json_schema": {
    ///                         "type": "object",
    ///                         "properties": {
    ///                             "city": {"type": "string"},
    ///                             "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
    ///                         },
    ///                         "required": ["city"]
    ///                     }
    ///                 },
    ///                 "end": "</function>"
    ///             }
    ///         ]
    ///     }
    /// });
    ///
    /// let grammar = Grammar::from_structural_tag(&structural_tag.to_string()).unwrap();
    /// assert!(!grammar.is_null());
    /// ```
    pub fn from_structural_tag(structural_tag_json: &str) -> Result<Self> {
        let structural_tag_json_cstring = CString::new(structural_tag_json)
            .expect("Failed to convert structural_tag_json to CString");
        let structural_tag_json_ptr = structural_tag_json_cstring.as_ptr();

        // We use separate variables to capture success flag and error message
        let mut success: i32 = 0;
        let mut error_msg_ptr: *mut std::os::raw::c_char = std::ptr::null_mut();

        let grammar = cpp!(unsafe [
            structural_tag_json_ptr as "const char*",
            mut success as "int",
            mut error_msg_ptr as "char*"
        ] -> Grammar as "xgrammar::Grammar" {
            std::string structural_tag_json_str(structural_tag_json_ptr);
            auto result = xgrammar::Grammar::FromStructuralTag(structural_tag_json_str);

            // Check if result holds a Grammar or an error
            if (std::holds_alternative<xgrammar::Grammar>(result)) {
                success = 1;
                error_msg_ptr = nullptr;
                return std::get<xgrammar::Grammar>(result);
            } else {
                success = 0;
                auto error = std::get<xgrammar::StructuralTagError>(result);

                // Extract error message from the variant
                std::string error_msg;
                std::visit([&error_msg](auto&& err) {
                    error_msg = err.what();
                }, error);

                // Allocate and copy error message
                error_msg_ptr = strdup(error_msg.c_str());

                // Return a null grammar
                return xgrammar::Grammar(xgrammar::NullObj{});
            }
        });

        if success == 1 {
            Ok(grammar)
        } else {
            let error_msg = if !error_msg_ptr.is_null() {
                let c_str = unsafe { CString::from_raw(error_msg_ptr) };
                c_str.to_string_lossy().into_owned()
            } else {
                "Unknown error".to_string()
            };
            Err(XGrammarErr::InvalidStructuralTag(error_msg))
        }
    }

    /// Get a grammar for standard JSON format.
    ///
    /// This method returns a pre-built grammar that matches any valid JSON according
    /// to the JSON specification, without schema constraints. It's useful as a starting
    /// point for JSON generation or when you need to accept any valid JSON structure.
    ///
    /// # Returns
    /// * A Grammar object that matches standard JSON format
    ///
    /// # Example
    /// ```
    /// # use xgrammar::Grammar;
    /// let json_grammar = Grammar::builtin_json_grammar();
    /// assert!(!json_grammar.is_null());
    /// ```
    pub fn builtin_json_grammar() -> Self {
        cpp!(unsafe [] -> Grammar as "xgrammar::Grammar" {
            return xgrammar::Grammar::BuiltinJSONGrammar();
        })
    }

    /// Create a grammar that matches any of the provided grammars.
    ///
    /// This method combines multiple grammars using a union operation, creating a new grammar
    /// that accepts input matching any of the input grammars. This is equivalent to the `|`
    /// (OR) operator in regular expressions.
    ///
    /// # Arguments
    /// * `grammars` - A slice of Grammar objects to combine
    ///
    /// # Returns
    /// * A new Grammar that matches if any of the input grammars match
    ///
    /// # Example
    /// ```
    /// # use xgrammar::Grammar;
    /// let grammar1 = Grammar::from_regex(r"[0-9]+", Some(false)).unwrap();
    /// let grammar2 = Grammar::from_regex(r"[a-z]+", Some(false)).unwrap();
    /// let union_grammar = Grammar::union(&[grammar1, grammar2]);
    /// assert!(!union_grammar.is_null());
    /// ```
    pub fn union(grammars: &[Grammar]) -> Self {
        let grammars_ptr = grammars.as_ptr();
        let num_grammars = grammars.len();
        cpp!(unsafe [
            grammars_ptr as "const xgrammar::Grammar*",
            num_grammars as "size_t"
        ] -> Grammar as "xgrammar::Grammar" {
            std::vector<xgrammar::Grammar> grammars_vec;
            grammars_vec.reserve(num_grammars);
            for (size_t i = 0; i < num_grammars; ++i) {
                grammars_vec.push_back(grammars_ptr[i]);
            }
            return xgrammar::Grammar::Union(grammars_vec);
        })
    }

    /// Create a grammar that matches the concatenation of the provided grammars.
    ///
    /// This method combines multiple grammars in sequence, creating a new grammar that requires
    /// input to match all grammars in order. This is equivalent to the `+` (concatenation)
    /// operator in formal language theory.
    ///
    /// # Arguments
    /// * `grammars` - A slice of Grammar objects to concatenate in order
    ///
    /// # Returns
    /// * A new Grammar that matches the sequential combination of all input grammars
    ///
    /// # Example
    /// ```
    /// # use xgrammar::Grammar;
    /// let greeting = Grammar::from_regex(r"Hello", Some(false)).unwrap();
    /// let space = Grammar::from_regex(r" ", Some(false)).unwrap();
    /// let name = Grammar::from_regex(r"[A-Z][a-z]+", Some(false)).unwrap();
    /// let concat_grammar = Grammar::concat(&[greeting, space, name]);
    /// assert!(!concat_grammar.is_null());
    /// ```
    pub fn concat(grammars: &[Grammar]) -> Self {
        let grammars_ptr = grammars.as_ptr();
        let num_grammars = grammars.len();
        cpp!(unsafe [
            grammars_ptr as "const xgrammar::Grammar*",
            num_grammars as "size_t"
        ] -> Grammar as "xgrammar::Grammar" {
            std::vector<xgrammar::Grammar> grammars_vec;
            grammars_vec.reserve(num_grammars);
            for (size_t i = 0; i < num_grammars; ++i) {
                grammars_vec.push_back(grammars_ptr[i]);
            }
            return xgrammar::Grammar::Concat(grammars_vec);
        })
    }

    /// Check if the grammar object is null.
    ///
    /// A null grammar typically indicates an uninitialized or invalid grammar state.
    /// This can occur when grammar construction fails or when working with default values.
    ///
    /// # Returns
    /// * `true` if the grammar is null (invalid/uninitialized)
    /// * `false` if the grammar is valid
    ///
    /// # Example
    /// ```
    /// # use xgrammar::Grammar;
    /// let grammar = Grammar::builtin_json_grammar();
    /// assert!(!grammar.is_null());
    /// ```
    pub fn is_null(&self) -> bool {
        cpp!(unsafe [self as "const xgrammar::Grammar*"] -> bool as "bool" {
            return self->IsNull();
        })
    }
}

impl GrammarMatcher {
    /// Create a GrammarMatcher from a compiled grammar.
    /// # Arguments
    /// * `compiled_grammar` - The compiled grammar to use
    pub fn new(compiled_grammar: &CompiledGrammar) -> Self {
        Self::with(compiled_grammar, None, Some(true), None)
    }

    /// Create a GrammarMatcher from a compiled grammar.
    /// # Arguments
    /// * `compiled_grammar` - The compiled grammar to use
    /// * `override_stop_tokens` - Optional list of token ids to override the default stop tokens
    /// * `terminate_without_stop_token` - Whether to terminate the matcher without accepting a stop token.
    /// * `max_rollback_tokens` - Deprecated. You don't need to set it and it's always unlimited (-1).
    ///   The new Earley parser significantly reduces the number of states, so we can allow
    ///   unlimited rollback. The maximum number of rollback tokens allowed. The rollback operation
    ///   is useful for jump-forward decoding and speculative decoding.
    pub fn with(
        compiled_grammar: &CompiledGrammar,
        override_stop_tokens: Option<&[i32]>,
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
    ///   a DLTensor with shape (tokenizer.GetVocabSize() + 31) / 32, and dtype int32.
    /// * `index` - The index of the bitmask to fill. If None, the first bitmask is filled.
    /// * `debug_print` - If true, print debug information.
    ///
    /// # Returns
    /// * `Ok(bool)` - Whether the bitmask need to be applied (not all-true).
    /// * `Err(XGrammarErr)` - Error if the operation fails (e.g., matcher terminated, invalid bitmask).
    ///
    /// # Errors
    /// * Returns error if the matcher has terminated after accepting the stop token
    /// * Returns error if the bitmask has invalid dtype, shape, or device type
    pub fn fill_next_token_bitmask(
        &mut self,
        next_token_bitmask: &mut DLTensor,
        index: Option<usize>,
        debug_print: Option<bool>,
    ) -> Result<bool> {
        let dl_tensor = next_token_bitmask.dl_tensor();
        let index = index.unwrap_or(0) as i32;
        let debug_print = debug_print.unwrap_or(false);

        let result = cpp!(unsafe [self as "xgrammar::GrammarMatcher*", dl_tensor as "DLTensor*", index as "int32_t", debug_print as "bool"] -> MatcherResult as "MatcherResult" {
            try {
                bool value = self->FillNextTokenBitmask(dl_tensor, index, debug_print);
                return {true, value, nullptr};
            } catch (const std::exception& e) {
                return {false, false, strdup(e.what())};
            }
        });

        result.into()
    }

    /// Rollback the matcher to a previous state.
    ///
    /// # Arguments
    /// * `num_tokens` - The number of tokens to rollback. It cannot exceed the current number of
    ///   steps, nor can it exceed the specified maximum number of rollback tokens.
    ///
    /// # Returns
    /// * `Ok(())` - If the rollback succeeds
    /// * `Err(XGrammarErr)` - Error if the rollback fails (e.g., num_tokens exceeds history)
    ///
    /// # Errors
    /// * Returns error if num_tokens exceeds the number of saved history steps
    pub fn rollback(&mut self, num_tokens: Option<i32>) -> Result<()> {
        let num_tokens = num_tokens.unwrap_or(1);

        let result = cpp!(unsafe [self as "xgrammar::GrammarMatcher*", num_tokens as "int"] -> MatcherResult as "MatcherResult" {
            try {
                self->Rollback(num_tokens);
                return {true, false, nullptr};
            } catch (const std::exception& e) {
                return {false, false, strdup(e.what())};
            }
        });

        result.into()
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

    pub fn get_stop_token_ids(&self) -> Vec<i32> {
        cpp!(unsafe [self as "const xgrammar::GrammarMatcher*"] -> Vec<i32> as "std::vector<int>" {
            return self->GetStopTokenIds();
        })
    }

    /// Reset the matcher to the initial state.
    pub fn reset(&mut self) {
        cpp!(unsafe [self as "xgrammar::GrammarMatcher*"] {
            self->Reset();
        })
    }
}

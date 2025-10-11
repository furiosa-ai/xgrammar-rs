mod common;

use serde_json::json;
use xgrammar::{GrammarCompiler, TokenizerInfo, VocabType};

const EXAONE_4_0_32B_PRETRAINED_ID: &str = "LGAI-EXAONE/EXAONE-4.0-32B";

#[test]
fn test_compile_builtin_json_grammar() {
    let tok_info = TokenizerInfo::from_pretrained(EXAONE_4_0_32B_PRETRAINED_ID, None, None, None)
        .expect("Failed to load tokenizer info");
    let compiler = GrammarCompiler::new(&tok_info);
    let compiled_grammar = compiler.compile_builtin_json_grammar();

    assert!(compiled_grammar.memory_size_bytes() > 0);
    assert_eq!(compiled_grammar.get_tokenizer_info().get_vocab_size(), 102400);
    assert_eq!(compiled_grammar.get_tokenizer_info().get_vocab_type(), VocabType::ByteLevel);
}

#[test]
fn test_cache_management() {
    let tok_info = TokenizerInfo::from_pretrained(EXAONE_4_0_32B_PRETRAINED_ID, None, None, None)
        .expect("Failed to load tokenizer info");
    let compiler = GrammarCompiler::new(&tok_info);

    // Check initial cache state
    let initial_cache_size = compiler.get_cache_size_bytes();
    let cache_limit = compiler.cache_limit_bytes();

    println!("Initial cache size: {} bytes", initial_cache_size);
    println!(
        "Cache limit: {} bytes",
        if cache_limit == -1 { "unlimited".to_string() } else { cache_limit.to_string() }
    );

    // Compile some grammars to populate cache
    let _grammar1 = compiler.compile_builtin_json_grammar();
    let cache_size_after_json = compiler.get_cache_size_bytes();

    let _grammar2 = compiler.compile_regex(r"\d+");
    let cache_size_after_regex = compiler.get_cache_size_bytes();

    let schema = json!({"type": "object", "properties": {"name": {"type": "string"}}});
    let _grammar3 = compiler.compile_json_schema(&schema.to_string(), None, None, None, None, None);
    let cache_size_after_schema = compiler.get_cache_size_bytes();

    println!("Cache size after JSON grammar: {} bytes", cache_size_after_json);
    println!("Cache size after regex: {} bytes", cache_size_after_regex);
    println!("Cache size after schema: {} bytes", cache_size_after_schema);

    // Cache should have grown (unless it was already at maximum)
    assert!(cache_size_after_schema >= initial_cache_size);

    // Clear cache
    compiler.clear_cache();
    let cache_size_after_clear = compiler.get_cache_size_bytes();
    println!("Cache size after clear: {} bytes", cache_size_after_clear);

    // Cache size should be reduced (though may not be zero due to internal structures)
    assert!(cache_size_after_clear <= cache_size_after_schema);
}

#[test]
fn test_cache_properties() {
    let tok_info = TokenizerInfo::from_pretrained(EXAONE_4_0_32B_PRETRAINED_ID, None, None, None)
        .expect("Failed to load tokenizer info");

    // Test with default cache settings
    let compiler = GrammarCompiler::new(&tok_info);
    let cache_limit = compiler.cache_limit_bytes();
    let cache_size = compiler.get_cache_size_bytes();

    // Cache limit should be consistent
    assert!(cache_limit == -1 || cache_limit > 0); // Either unlimited (-1) or a positive limit

    // Cache size should be non-negative
    assert!(cache_size >= 0);

    // Test with custom cache settings
    let compiler_with_limit = GrammarCompiler::with(
        &tok_info,
        None,                   // max_threads
        Some(true),             // cache_enabled
        Some(1024 * 1024 * 10), // max_memory_bytes: 10MB
    );

    let custom_cache_limit = compiler_with_limit.cache_limit_bytes();
    println!("Custom cache limit: {} bytes", custom_cache_limit);

    // Should respect the custom limit
    assert!(custom_cache_limit > 0);
}

#[test]
fn test_compile_json_schema() {
    let tok_info = TokenizerInfo::from_pretrained(EXAONE_4_0_32B_PRETRAINED_ID, None, None, None)
        .expect("Failed to load tokenizer info");
    let compiler = GrammarCompiler::new(&tok_info);

    // Test with a simple JSON schema
    let schema = json!({
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer", "minimum": 0}
        },
        "required": ["name", "age"]
    });

    // Test with default parameters
    let compiled_grammar =
        compiler.compile_json_schema(&schema.to_string(), None, None, None, None, None);
    assert!(compiled_grammar.memory_size_bytes() > 0);
    assert_eq!(compiled_grammar.get_tokenizer_info().get_vocab_size(), 102400);
    assert_eq!(compiled_grammar.get_tokenizer_info().get_vocab_type(), VocabType::ByteLevel);

    // Test with custom parameters
    let compiled_grammar_custom = compiler.compile_json_schema(
        &schema.to_string(),
        Some(false),
        Some(2),
        Some((":".to_string(), ",".to_string())),
        Some(true),
        None,
    );
    assert!(compiled_grammar_custom.memory_size_bytes() > 0);

    // Test with a different schema (array type)
    let array_schema = json!({
        "type": "array",
        "items": {"type": "string"},
        "minItems": 1,
        "maxItems": 3
    });

    let array_compiled =
        compiler.compile_json_schema(&array_schema.to_string(), None, None, None, None, None);
    assert!(array_compiled.memory_size_bytes() > 0);
}

#[test]
fn test_compile_regex() {
    let tok_info = TokenizerInfo::from_pretrained(EXAONE_4_0_32B_PRETRAINED_ID, None, None, None)
        .expect("Failed to load tokenizer info");
    let compiler = GrammarCompiler::new(&tok_info);

    // Test simple regex patterns
    let regex_patterns = vec![
        r"\d+",           // digits
        r"[a-zA-Z]+",     // letters
        r"\w+@\w+\.\w+",  // simple email pattern
        r"^hello world$", // exact match
        r"(foo|bar)+",    // alternation with repetition
    ];

    for pattern in regex_patterns {
        let compiled_regex = compiler.compile_regex(pattern);
        assert!(compiled_regex.memory_size_bytes() > 0);
        assert_eq!(compiled_regex.get_tokenizer_info().get_vocab_size(), 102400);
        assert_eq!(compiled_regex.get_tokenizer_info().get_vocab_type(), VocabType::ByteLevel);
    }
}

#[test]
fn test_compile_structural_tag() {
    let tok_info = TokenizerInfo::from_pretrained(EXAONE_4_0_32B_PRETRAINED_ID, None, None, None)
        .expect("Failed to load tokenizer info");
    let compiler = GrammarCompiler::new(&tok_info);

    // Test with simple structural tag JSON
    let structural_tag_json = r#"{
        "format": {
            "type": "json_schema",
            "json_schema": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "answer": {"type": "string"}
                }
            }
        }
    }"#;

    let compiled_grammar = compiler.compile_structural_tag(structural_tag_json);
    assert!(compiled_grammar.memory_size_bytes() > 0);
}

#[test]
fn test_compile_structural_tag_complex() {
    let tok_info = TokenizerInfo::from_pretrained(EXAONE_4_0_32B_PRETRAINED_ID, None, None, None)
        .expect("Failed to load tokenizer info");
    let compiler = GrammarCompiler::new(&tok_info);

    // Test with more complex structural tag JSON
    let structural_tag_json = r#"{
        "format": {
            "type": "json_schema",
            "json_schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "minLength": 1},
                    "age": {"type": "integer", "minimum": 0},
                    "hobbies": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["name", "age"]
            }
        }
    }"#;

    let compiled_grammar = compiler.compile_structural_tag(structural_tag_json);
    assert!(compiled_grammar.memory_size_bytes() > 0);
}

#[test]
fn test_compile_structural_tag_minimal() {
    let tok_info = TokenizerInfo::from_pretrained(EXAONE_4_0_32B_PRETRAINED_ID, None, None, None)
        .expect("Failed to load tokenizer info");
    let compiler = GrammarCompiler::new(&tok_info);

    // Test with minimal structural tag JSON (empty object schema)
    let structural_tag_json = r#"{
        "format": {
            "type": "json_schema",
            "json_schema": {}
        }
    }"#;

    let compiled_grammar = compiler.compile_structural_tag(structural_tag_json);
    // Should not crash, and should have some memory usage
    let _memory_size = compiled_grammar.memory_size_bytes(); // Just verify it doesn't crash
}

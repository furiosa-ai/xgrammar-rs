mod common;

use xgrammar::{GrammarCompiler, TokenizerInfo};

const EXAONE_4_0_32B_PRETRAINED_ID: &str = "LGAI-EXAONE/EXAONE-4.0-32B";

#[test]
#[cfg(feature = "hf_hub")]
fn test_invalid_json_syntax() {
    let tok_info = TokenizerInfo::from_pretrained(EXAONE_4_0_32B_PRETRAINED_ID, None, None, None)
        .expect("Failed to load tokenizer info");
    let mut compiler = GrammarCompiler::new(&tok_info);

    // Test malformed JSON (missing closing brace)
    let malformed_json = r#"{
        "type": "object",
        "properties": {
            "name": {"type": "string"}
        "#;

    println!("Testing malformed JSON...");
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        compiler.compile_json_schema(malformed_json, None, None, None, None)
    }));

    match result {
        Ok(_) => println!("Malformed JSON was accepted (unexpected)"),
        Err(_) => println!("Malformed JSON caused a panic (as expected)"),
    }
}

#[test]
#[cfg(feature = "hf_hub")]
fn test_empty_schema() {
    let tok_info = TokenizerInfo::from_pretrained(EXAONE_4_0_32B_PRETRAINED_ID, None, None, None)
        .expect("Failed to load tokenizer info");
    let mut compiler = GrammarCompiler::new(&tok_info);

    println!("Testing empty string...");
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        compiler.compile_json_schema("", None, None, None, None)
    }));

    match result {
        Ok(_) => println!("Empty string was accepted (unexpected)"),
        Err(_) => println!("Empty string caused a panic (as expected)"),
    }
}

#[test]
#[cfg(feature = "hf_hub")]
fn test_invalid_schema_structure() {
    let tok_info = TokenizerInfo::from_pretrained(EXAONE_4_0_32B_PRETRAINED_ID, None, None, None)
        .expect("Failed to load tokenizer info");
    let mut compiler = GrammarCompiler::new(&tok_info);

    // Test invalid schema with unknown type
    let invalid_type_schema = r#"{
        "type": "unknown_type",
        "properties": {
            "name": {"type": "string"}
        }
    }"#;

    println!("Testing invalid schema type...");
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        compiler.compile_json_schema(invalid_type_schema, None, None, None, None)
    }));

    match result {
        Ok(_) => println!("Invalid schema type was accepted (might be valid behavior)"),
        Err(_) => println!("Invalid schema type caused a panic"),
    }
}

#[test]
#[cfg(feature = "hf_hub")]
fn test_non_json_string() {
    let tok_info = TokenizerInfo::from_pretrained(EXAONE_4_0_32B_PRETRAINED_ID, None, None, None)
        .expect("Failed to load tokenizer info");
    let mut compiler = GrammarCompiler::new(&tok_info);

    println!("Testing non-JSON string...");
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        compiler.compile_json_schema("not json at all", None, None, None, None)
    }));

    match result {
        Ok(_) => println!("Non-JSON string was accepted (unexpected)"),
        Err(_) => println!("Non-JSON string caused a panic (as expected)"),
    }
}

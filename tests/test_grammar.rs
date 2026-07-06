mod common;

use serde_json::json;
use xgrammar::{Grammar, XGrammarErr};

#[test]
fn test_grammar_from_ebnf() {
    let ebnf = r#"
        root ::= "a" "b"
    "#;
    let grammar = Grammar::from_ebnf(ebnf, None).unwrap();
    assert!(!grammar.is_null());
}

#[test]
fn test_grammar_from_json_schema() {
    let schema = json!({
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        }
    });
    let grammar =
        Grammar::from_json_schema(&schema.to_string(), None, None, None, None, None, None).unwrap();
    assert!(!grammar.is_null());
}

#[test]
fn test_grammar_from_regex() {
    let regex = "[a-z]+";
    let grammar = Grammar::from_regex(regex, None).unwrap();
    assert!(!grammar.is_null());
}

#[test]
fn test_grammar_from_structural_tag_simple() {
    // Test with minimal valid structural tag JSON
    let structural_tag_json = json!({
        "format": {
            "type": "json_schema",
            "json_schema": {}
        }
    });

    let result = Grammar::from_structural_tag(&structural_tag_json.to_string(), None);
    assert!(result.is_ok());
    let grammar = result.unwrap();
    assert!(!grammar.is_null());
}

#[test]
fn test_from_structural_tag_errors() {
    // Missing format field using json! macro
    let missing_format = json!({
        "type": "structural_tag"
    });
    let result = Grammar::from_structural_tag(&missing_format.to_string(), None);
    let Err(XGrammarErr::InvalidStructuralTag(err_msg)) = result else {
        panic!("Expected InvalidStructuralTag");
    };
    assert_eq!(&err_msg, "Invalid structural tag error: Structural tag must have a format field");
}

#[test]
fn test_from_structural_tag_invalid_json() {
    let result = Grammar::from_structural_tag("{ not json", None);
    let Err(XGrammarErr::InvalidJson(err_msg)) = result else {
        panic!("Expected InvalidJson");
    };
    assert!(err_msg.contains("Invalid JSON error"), "unexpected message: {err_msg}");
}

#[test]
fn test_builtin_json_grammar() {
    let grammar = Grammar::builtin_json_grammar();
    assert!(!grammar.is_null());
}

#[test]
fn test_grammar_union() {
    let grammar1 = Grammar::from_regex("a", None).unwrap();
    let grammar2 = Grammar::from_regex("b", None).unwrap();
    let union_grammar = Grammar::union(&[grammar1, grammar2]);
    assert!(!union_grammar.is_null());
}

#[test]
fn test_grammar_concat() {
    let grammar1 = Grammar::from_regex("a", None).unwrap();
    let grammar2 = Grammar::from_regex("b", None).unwrap();
    let concat_grammar = Grammar::concat(&[grammar1, grammar2]);
    assert!(!concat_grammar.is_null());
}

#[test]
fn test_grammar_from_ebnf_error() {
    let invalid_ebnf = r#"root ::= "unterminated string"#;
    let Err(XGrammarErr::InvalidGrammar(err_msg)) = Grammar::from_ebnf(invalid_ebnf, None) else {
        panic!("Expected grammar creation to fail, but it succeeded");
    };

    assert!(err_msg.contains("EBNF lexer error at line 1, column 30: Expect \" in string literal"));
}

#[test]
fn test_grammar_from_json_schema_error() {
    let invalid_json = "{ invalid json }";
    let Err(XGrammarErr::InvalidGrammar(err_msg)) =
        Grammar::from_json_schema(invalid_json, None, None, None, None, None, None)
    else {
        panic!("Expected grammar creation to fail, but it succeeded");
    };

    assert!(err_msg.contains("Failed to parse JSON: syntax error"));
}

#[test]
fn test_grammar_from_regex_error() {
    // Unclosed bracket
    let invalid_regex = "[";
    let Err(XGrammarErr::InvalidGrammar(err_msg)) = Grammar::from_regex(invalid_regex, None) else {
        panic!("Expected grammar creation to fail, but it succeeded");
    };

    assert!(err_msg.contains("Regex parsing error at position 2: Unclosed '['"));
}

/// The rendered (`Display`) output of C++-originating errors is the upstream
/// message verbatim — the binding adds no prefix of its own, and the
/// upstream type prefix is not duplicated.
#[test]
fn test_error_display_is_upstream_message() {
    // Typed error (InvalidStructuralTag): exact upstream `what()`.
    let missing_format = json!({"type": "structural_tag"});
    let err =
        Grammar::from_structural_tag(&missing_format.to_string(), None).map(|_| ()).unwrap_err();
    assert_eq!(
        err.to_string(),
        "Invalid structural tag error: Structural tag must have a format field"
    );

    // Typed error (InvalidJson): the upstream type prefix appears exactly once.
    let err = Grammar::deserialize_json("not json").map(|_| ()).unwrap_err();
    let display = err.to_string();
    assert!(display.starts_with("Invalid JSON error: "), "unexpected display: {display}");
    assert_eq!(display.matches("Invalid JSON error").count(), 1, "duplicated prefix: {display}");

    // Untyped fallback (InvalidGrammar): upstream message verbatim.
    let err = Grammar::from_ebnf(r#"root ::= "unterminated string"#, None).map(|_| ()).unwrap_err();
    assert!(
        err.to_string().contains("EBNF lexer error at line 1, column 30"),
        "unexpected display: {err}"
    );
}

#[test]
fn test_grammar_serialize_roundtrip() {
    let ebnf = r#"
        root ::= "a" name
        name ::= [A-Z][a-z]+
    "#;
    let grammar = Grammar::from_ebnf(ebnf, None).unwrap();
    let json = grammar.serialize_json();
    assert!(!json.is_empty());

    let restored = Grammar::deserialize_json(&json).expect("deserialization should succeed");
    assert!(!restored.is_null());
    // Serializing the restored grammar must reproduce the same JSON.
    assert_eq!(json, restored.serialize_json());
}

#[test]
fn test_grammar_deserialize_garbage() {
    let Err(XGrammarErr::InvalidJson(err_msg)) = Grammar::deserialize_json("not json") else {
        panic!("Expected InvalidJson");
    };
    assert!(err_msg.contains("Invalid JSON error"), "unexpected message: {err_msg}");
}

#[test]
fn test_grammar_deserialize_missing_version() {
    // A well-formed JSON object without the serialization version marker.
    let result = Grammar::deserialize_json("{}");
    match result {
        Err(XGrammarErr::DeserializeVersion(_)) | Err(XGrammarErr::DeserializeFormat(_)) => {}
        _ => panic!("Expected DeserializeVersion or DeserializeFormat"),
    }
}

#[test]
fn test_grammar_deserialize_wrong_version() {
    let grammar = Grammar::builtin_json_grammar();
    let json = grammar.serialize_json();

    // Tamper with the serialization version marker.
    let mut value: serde_json::Value = serde_json::from_str(&json).unwrap();
    let obj = value.as_object_mut().expect("serialized grammar should be a JSON object");
    assert!(obj.contains_key("__VERSION__"), "expected a __VERSION__ marker");
    obj.insert("__VERSION__".to_string(), json!("v0"));

    let Err(XGrammarErr::DeserializeVersion(err_msg)) =
        Grammar::deserialize_json(&value.to_string())
    else {
        panic!("Expected DeserializeVersion");
    };
    assert!(err_msg.contains("Deserialize version error"), "unexpected message: {err_msg}");
}

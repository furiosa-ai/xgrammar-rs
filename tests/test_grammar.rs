mod common;

use serde_json::json;
use xgrammar::Grammar;

#[test]
fn test_grammar_from_ebnf() {
    let ebnf = r#"
        root ::= "a" "b"
    "#;
    let grammar = Grammar::from_ebnf(ebnf, None);
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
        Grammar::from_json_schema(&schema.to_string(), None, None, None, None, None, None);
    assert!(!grammar.is_null());
}

#[test]
fn test_grammar_from_regex() {
    let regex = "[a-z]+";
    let grammar = Grammar::from_regex(regex, None);
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

    let result = Grammar::from_structural_tag(&structural_tag_json.to_string());
    assert!(result.is_ok());
    let grammar = result.unwrap();
    assert!(!grammar.is_null());
}

#[test]
fn test_from_structural_tag_errors() {
    // Test 1: Invalid JSON syntax
    let invalid_json = "not a json";
    let result = Grammar::from_structural_tag(invalid_json);
    assert!(result.is_err());
    if let Err(err) = result {
        let err_msg = err.to_string();
        assert!(err_msg.contains("Invalid JSON error"));
        assert!(err_msg.contains("Failed to parse JSON"));
    }

    // Test 2: Missing format field using json! macro
    let missing_format = json!({
        "type": "structural_tag"
    });
    let result = Grammar::from_structural_tag(&missing_format.to_string());
    assert!(result.is_err());
    if let Err(err) = result {
        let err_msg = err.to_string();
        assert!(err_msg.contains("Invalid structural tag error"));
        assert!(err_msg.contains("Structural tag must have a format field"));
    }

    // Test 3: Invalid format type using json! macro
    let invalid_format_type = json!({
        "format": {
            "type": "invalid_type"
        }
    });
    let result = Grammar::from_structural_tag(&invalid_format_type.to_string());
    assert!(result.is_err());
    if let Err(err) = result {
        let err_msg = err.to_string();
        assert!(err_msg.contains("invalid structural tag"));
    }

    // Test 4: Missing json_schema field in json_schema format using json! macro
    let missing_json_schema = json!({
        "format": {
            "type": "json_schema"
        }
    });
    let result = Grammar::from_structural_tag(&missing_json_schema.to_string());
    assert!(result.is_err());
    if let Err(err) = result {
        let err_msg = err.to_string();
        assert!(err_msg.contains("Invalid structural tag error"));
        assert!(err_msg.contains("JSON schema format must have a json_schema field"));
    }
}

#[test]
fn test_builtin_json_grammar() {
    let grammar = Grammar::builtin_json_grammar();
    assert!(!grammar.is_null());
}

#[test]
fn test_grammar_union() {
    let grammar1 = Grammar::from_regex("a", None);
    let grammar2 = Grammar::from_regex("b", None);
    let union_grammar = Grammar::union(&[grammar1, grammar2]);
    assert!(!union_grammar.is_null());
}

#[test]
fn test_grammar_concat() {
    let grammar1 = Grammar::from_regex("a", None);
    let grammar2 = Grammar::from_regex("b", None);
    let concat_grammar = Grammar::concat(&[grammar1, grammar2]);
    assert!(!concat_grammar.is_null());
}

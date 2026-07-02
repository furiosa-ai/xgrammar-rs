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
    let Err(XGrammarErr::InvalidGrammar(err_msg)) = result else {
        panic!("Expected grammar creation to fail, but it succeeded");
    };
    assert_eq!(&err_msg, "Invalid structural tag error: Structural tag must have a format field");
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
        Grammar::from_json_schema(invalid_json, None, None, None, None, None, None, None)
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

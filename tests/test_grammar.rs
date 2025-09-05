mod common;

use xgrammar::{Grammar, StructuralTagItem};

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
    let schema = r#"{
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        }
    }"#;
    let grammar = Grammar::from_json_schema(schema, None, None, None, None, None);
    assert!(!grammar.is_null());
}

#[test]
fn test_grammar_from_regex() {
    let regex = "[a-z]+";
    let grammar = Grammar::from_regex(regex, None);
    assert!(!grammar.is_null());
}

#[test]
fn test_grammar_from_structural_tag() {
    let tags = vec![StructuralTagItem::new(
        "<start>".to_string(),
        r#"{"type": "string"}"#.to_string(),
        "<end>".to_string(),
    )];
    let triggers = vec!["<start>".to_string()];
    let grammar = Grammar::from_structural_tag(&tags, &triggers);
    assert!(!grammar.is_null());
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

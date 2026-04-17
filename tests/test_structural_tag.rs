use serde_json::json;
use xgrammar::Grammar;

// ============================================================================
// Basic Format Types
// ============================================================================

/// Test const_string format: forces exact string match
#[test]
fn test_const_string_format() {
    let structural_tag = json!({
        "type": "structural_tag",
        "format": {
            "type": "const_string",
            "value": "Hello, World!"
        }
    });

    let grammar = Grammar::from_structural_tag(&structural_tag.to_string(), None);
    assert!(grammar.is_ok());
    assert!(!grammar.unwrap().is_null());
}

/// Test json_schema format: validates JSON structure
#[test]
fn test_json_schema_format() {
    let structural_tag = json!({
        "type": "structural_tag",
        "format": {
            "type": "json_schema",
            "json_schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"}
                },
                "required": ["name"]
            }
        }
    });

    let grammar = Grammar::from_structural_tag(&structural_tag.to_string(), None);
    assert!(grammar.is_ok());
    assert!(!grammar.unwrap().is_null());
}

/// Test regex format: matches regular expression patterns
#[test]
fn test_regex_format() {
    let structural_tag = json!({
        "type": "structural_tag",
        "format": {
            "type": "regex",
            "pattern": r"[0-9]{3}-[0-9]{4}"
        }
    });

    let grammar = Grammar::from_structural_tag(&structural_tag.to_string(), None);
    assert!(grammar.is_ok());
    assert!(!grammar.unwrap().is_null());
}

/// Test grammar format: uses EBNF grammar specification
#[test]
fn test_grammar_format() {
    let structural_tag = json!({
        "type": "structural_tag",
        "format": {
            "type": "grammar",
            "grammar": "root ::= \"Hello!\" number\nnumber ::= [0-9] | [0-9] number"
        }
    });

    let grammar = Grammar::from_structural_tag(&structural_tag.to_string(), None);
    assert!(grammar.is_ok());
    assert!(!grammar.unwrap().is_null());
}

/// Test any_text format: allows any text (must be used with tag to detect end)
#[test]
fn test_any_text_format() {
    let structural_tag = json!({
        "type": "structural_tag",
        "format": {
            "type": "tag",
            "begin": "<text>",
            "content": {"type": "any_text"},
            "end": "</text>"
        }
    });

    let grammar = Grammar::from_structural_tag(&structural_tag.to_string(), None);
    assert!(grammar.is_ok());
    assert!(!grammar.unwrap().is_null());
}

// ============================================================================
// Composite Format Types
// ============================================================================

/// Test sequence format: matches elements in order
#[test]
fn test_sequence_format() {
    let structural_tag = json!({
        "type": "structural_tag",
        "format": {
            "type": "sequence",
            "elements": [
                {"type": "const_string", "value": "START"},
                {"type": "json_schema", "json_schema": {"type": "number"}},
                {"type": "const_string", "value": "END"}
            ]
        }
    });

    let grammar = Grammar::from_structural_tag(&structural_tag.to_string(), None);
    assert!(grammar.is_ok());
    assert!(!grammar.unwrap().is_null());
}

/// Test or format: matches any of the provided elements
#[test]
fn test_or_format() {
    let structural_tag = json!({
        "type": "structural_tag",
        "format": {
            "type": "or",
            "elements": [
                {"type": "const_string", "value": "yes"},
                {"type": "const_string", "value": "no"}
            ]
        }
    });

    let grammar = Grammar::from_structural_tag(&structural_tag.to_string(), None);
    assert!(grammar.is_ok());
    assert!(!grammar.unwrap().is_null());
}

/// Test tag format: wraps content with begin/end markers
#[test]
fn test_tag_format() {
    let structural_tag = json!({
        "type": "structural_tag",
        "format": {
            "type": "tag",
            "begin": "<data>",
            "content": {"type": "json_schema", "json_schema": {"type": "string"}},
            "end": "</data>"
        }
    });

    let grammar = Grammar::from_structural_tag(&structural_tag.to_string(), None);
    assert!(grammar.is_ok());
    assert!(!grammar.unwrap().is_null());
}

// ============================================================================
// Advanced Format Types
// ============================================================================

/// Test triggered_tags format: dispatches to different tags based on triggers
#[test]
fn test_triggered_tags_format() {
    let structural_tag = json!({
        "type": "structural_tag",
        "format": {
            "type": "triggered_tags",
            "triggers": ["<function="],
            "tags": [
                {
                    "begin": "<function=get_weather>",
                    "content": {
                        "type": "json_schema",
                        "json_schema": {
                            "type": "object",
                            "properties": {
                                "city": {"type": "string"}
                            }
                        }
                    },
                    "end": "</function>"
                },
                {
                    "begin": "<function=get_time>",
                    "content": {
                        "type": "json_schema",
                        "json_schema": {
                            "type": "object",
                            "properties": {
                                "timezone": {"type": "string"}
                            }
                        }
                    },
                    "end": "</function>"
                }
            ]
        }
    });

    let grammar = Grammar::from_structural_tag(&structural_tag.to_string(), None);
    assert!(grammar.is_ok());
    assert!(!grammar.unwrap().is_null());
}

/// Test triggered_tags with at_least_one option
#[test]
fn test_triggered_tags_at_least_one() {
    let structural_tag = json!({
        "type": "structural_tag",
        "format": {
            "type": "triggered_tags",
            "triggers": ["<tool="],
            "tags": [
                {
                    "begin": "<tool=search>",
                    "content": {"type": "json_schema", "json_schema": {"type": "object"}},
                    "end": "</tool>"
                }
            ],
            "at_least_one": true
        }
    });

    let grammar = Grammar::from_structural_tag(&structural_tag.to_string(), None);
    assert!(grammar.is_ok());
    assert!(!grammar.unwrap().is_null());
}

/// Test triggered_tags with stop_after_first option
#[test]
fn test_triggered_tags_stop_after_first() {
    let structural_tag = json!({
        "type": "structural_tag",
        "format": {
            "type": "triggered_tags",
            "triggers": ["<action="],
            "tags": [
                {
                    "begin": "<action=run>",
                    "content": {"type": "json_schema", "json_schema": {"type": "object"}},
                    "end": "</action>"
                }
            ],
            "stop_after_first": true
        }
    });

    let grammar = Grammar::from_structural_tag(&structural_tag.to_string(), None);
    assert!(grammar.is_ok());
    assert!(!grammar.unwrap().is_null());
}

/// Test tags_with_separator format: multiple tags separated by a delimiter
#[test]
fn test_tags_with_separator_format() {
    let structural_tag = json!({
        "type": "structural_tag",
        "format": {
            "type": "tags_with_separator",
            "tags": [
                {
                    "begin": "<item>",
                    "content": {"type": "json_schema", "json_schema": {"type": "string"}},
                    "end": "</item>"
                }
            ],
            "separator": ","
        }
    });

    let grammar = Grammar::from_structural_tag(&structural_tag.to_string(), None);
    assert!(grammar.is_ok());
    assert!(!grammar.unwrap().is_null());
}

/// Test tags_with_separator with at_least_one option
#[test]
fn test_tags_with_separator_at_least_one() {
    let structural_tag = json!({
        "type": "structural_tag",
        "format": {
            "type": "tags_with_separator",
            "tags": [
                {
                    "begin": "<entry>",
                    "content": {"type": "json_schema", "json_schema": {"type": "number"}},
                    "end": "</entry>"
                }
            ],
            "separator": ";",
            "at_least_one": true
        }
    });

    let grammar = Grammar::from_structural_tag(&structural_tag.to_string(), None);
    assert!(grammar.is_ok());
    assert!(!grammar.unwrap().is_null());
}

/// Test qwen_xml_parameter format: Qwen-style XML parameter format
#[test]
fn test_qwen_xml_parameter_format() {
    let structural_tag = json!({
        "type": "structural_tag",
        "format": {
            "type": "qwen_xml_parameter",
            "json_schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"}
                },
                "required": ["name", "age"]
            }
        }
    });

    let grammar = Grammar::from_structural_tag(&structural_tag.to_string(), None);
    assert!(grammar.is_ok());
    assert!(!grammar.unwrap().is_null());
}

// ============================================================================
// Complex Nested Examples
// ============================================================================

/// Test nested sequence with or
#[test]
fn test_nested_sequence_with_or() {
    let structural_tag = json!({
        "type": "structural_tag",
        "format": {
            "type": "sequence",
            "elements": [
                {"type": "const_string", "value": "Result: "},
                {
                    "type": "or",
                    "elements": [
                        {"type": "const_string", "value": "success"},
                        {"type": "const_string", "value": "failure"}
                    ]
                }
            ]
        }
    });

    let grammar = Grammar::from_structural_tag(&structural_tag.to_string(), None);
    assert!(grammar.is_ok());
    assert!(!grammar.unwrap().is_null());
}

/// Test tool calling with think tag
#[test]
fn test_tool_calling_with_think_tag() {
    let structural_tag = json!({
        "type": "structural_tag",
        "format": {
            "type": "sequence",
            "elements": [
                {
                    "type": "tag",
                    "begin": "<think>",
                    "content": {"type": "any_text"},
                    "end": "</think>"
                },
                {
                    "type": "triggered_tags",
                    "triggers": ["<function="],
                    "tags": [
                        {
                            "begin": "<function=calculate>",
                            "content": {
                                "type": "json_schema",
                                "json_schema": {
                                    "type": "object",
                                    "properties": {
                                        "expression": {"type": "string"}
                                    }
                                }
                            },
                            "end": "</function>"
                        }
                    ]
                }
            ]
        }
    });

    let grammar = Grammar::from_structural_tag(&structural_tag.to_string(), None);
    assert!(grammar.is_ok());
    assert!(!grammar.unwrap().is_null());
}

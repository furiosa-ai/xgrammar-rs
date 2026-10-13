use xgrammar::{CompiledGrammar, Grammar, GrammarCompiler, GrammarMatcher, TokenizerInfo};

const SCHEMA: &str = r#"{
    "type": "object",
    "properties": {
        "a": {"type": "integer"},
        "b": {
            "type": "object",
            "properties": {
                "x": {"type": "integer"},
                "y": {"type": "string"}
            },
            "required": ["x", "y"],
            "additionalProperties": false
        }
    },
    "required": ["a", "b"],
    "additionalProperties": false
}"#;

fn tokenizer_info() -> TokenizerInfo {
    TokenizerInfo::from_backend_str(
        r#"{"model":{"type":"WordLevel","vocab":{"a":0},"unk_token":"a"}}"#,
        None,
        vec![],
    )
    .expect("Failed to build tokenizer info")
}

fn assert_object_order(compiled: &CompiledGrammar, any_order: bool) {
    let mut matcher = GrammarMatcher::new(compiled);
    for (input, accepted) in [
        (r#"{"a":1,"b":{"x":2,"y":"ok"}}"#, true),
        (r#"{"b":{"x":2,"y":"ok"},"a":1}"#, any_order),
        (r#"{"a":1,"b":{"y":"ok","x":2}}"#, any_order),
        (r#"{"b":{"y":"ok","x":2},"a":1}"#, any_order),
        (r#"{"a":1,"b":{"x":2,"y":"ok"},"unknown":3}"#, false),
        (r#"{"a":1,"b":{"x":2,"y":"ok","unknown":3}}"#, false),
        (r#"{"a":"wrong","b":{"x":2,"y":"ok"}}"#, false),
        (r#"{"a":1,"b":{"x":2,"y":3}}"#, false),
    ] {
        matcher.reset();
        assert_eq!(
            matcher.accept_string(input, None),
            accepted,
            "any_order={any_order}, input: {input}"
        );
        if accepted {
            assert!(matcher.is_completed(), "input should complete the grammar: {input}");
        }
    }
}

#[test]
fn test_from_json_schema_any_order() {
    let tokenizer_info = tokenizer_info();
    let compiler = GrammarCompiler::new(&tokenizer_info);
    for any_order in [None, Some(false), Some(true)] {
        let grammar =
            Grammar::from_json_schema(SCHEMA, None, None, None, None, None, None, any_order)
                .expect("Failed to build JSON schema grammar");
        let compiled = compiler.compile_grammar(&grammar).expect("Failed to compile grammar");
        assert_object_order(&compiled, any_order.unwrap_or(false));
    }
}

#[test]
fn test_compile_json_schema_any_order() {
    let tokenizer_info = tokenizer_info();
    for cache_enabled in [false, true] {
        let compiler = GrammarCompiler::with(&tokenizer_info, None, Some(cache_enabled), None);
        // Alternate options on one compiler so cached ordered and unordered grammars
        // must remain distinct, including the default after compiling with Some(true).
        for any_order in [None, Some(true), Some(false), Some(true), None] {
            let compiled = compiler
                .compile_json_schema(SCHEMA, None, None, None, None, None, any_order)
                .expect("Failed to compile JSON schema");
            assert_object_order(&compiled, any_order.unwrap_or(false));
        }
    }
}

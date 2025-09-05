mod common;

use std::time::Instant;

use xgrammar::{Grammar, GrammarCompiler, GrammarMatcher, TokenizerInfo};

const GPT_OSS_20B_PRETRAINED_ID: &str = "openai/gpt-oss-20b";

/// Test CompiledGrammar and GrammarMatcher methods
// Ported from https://github.com/mlc-ai/xgrammar/blob/16e5298ed9b74fba1c8674b21996b0f47d95276d/tests/python/test_grammar_compiler.py#L17-L42
#[test]
fn test_compiled_grammar() {
    let grammar = Grammar::builtin_json_grammar();
    let tokenizer_info =
        TokenizerInfo::from_pretrained(GPT_OSS_20B_PRETRAINED_ID, None, None, None)
            .expect("Failed to load tokenizer info");
    let mut compiler = GrammarCompiler::new(&tokenizer_info);

    let time_start = Instant::now();
    let compiled_grammar = compiler.compile_grammar(&grammar);
    let time_end = Instant::now();
    println!("Time to get compiled grammar: {:?}", time_end.duration_since(time_start));

    let check_matcher = |mut matcher: GrammarMatcher| {
        assert!(!matcher.is_terminated());
        assert!(!matcher.accept_string("{ name: \"John\" }", None));
        assert!(matcher.accept_string("{\"name\": \"John\"}", None));
        assert!(matcher.is_terminated());
    };

    let time_start = Instant::now();
    let matcher_1 = GrammarMatcher::with(&compiled_grammar, None, Some(true), None);
    let time_end = Instant::now();
    tracing::info!("Time to init matcher 1: {:?}", time_end.duration_since(time_start));
    check_matcher(matcher_1);

    let time_start = Instant::now();
    let matcher_2 = GrammarMatcher::with(&compiled_grammar, None, Some(true), None);
    let time_end = Instant::now();
    tracing::info!("Time to init matcher 2: {:?}", time_end.duration_since(time_start));
    check_matcher(matcher_2);
}

/// Test GrammarMatcher with different threads
// Ported from https://github.com/mlc-ai/xgrammar/blob/16e5298ed9b74fba1c8674b21996b0f47d95276d/tests/python/test_grammar_compiler.py#L48-L83
fn do_test_grammar_compiler_json_test(max_threads: usize) {
    let tokenizer_info =
        TokenizerInfo::from_pretrained(GPT_OSS_20B_PRETRAINED_ID, None, None, None)
            .expect("Failed to load tokenizer info");

    let time_start = Instant::now();
    let mut grammar_compiler =
        GrammarCompiler::with(&tokenizer_info, Some(max_threads), None, None);
    let time_end = Instant::now();
    println!("Time to init cached grammar compiler: {:?}", time_end.duration_since(time_start));

    let check_matcher = |mut matcher: GrammarMatcher| {
        assert!(!matcher.is_terminated());
        assert!(!matcher.accept_string("{ name: \"John\" }", None));
        assert!(matcher.accept_string("{\"name\": \"John\"}", None));
        assert!(matcher.is_terminated());
    };

    let time_start = Instant::now();
    let compiled_grammar = grammar_compiler.compile_builtin_json_grammar();
    let time_end = Instant::now();
    println!("Time to get compiled grammar: {:?}", time_end.duration_since(time_start));
    let matcher = GrammarMatcher::with(&compiled_grammar, None, Some(true), None);
    check_matcher(matcher);

    let time_start = Instant::now();
    let compiled_grammar_again = grammar_compiler.compile_builtin_json_grammar();
    let time_end = Instant::now();
    println!("Time to get compiled grammar again: {:?}", time_end.duration_since(time_start));
    let matcher_again = GrammarMatcher::with(&compiled_grammar_again, None, Some(true), None);
    check_matcher(matcher_again);

    grammar_compiler.clear_cache();

    let time_start = Instant::now();
    let compiled_grammar_after_clear = grammar_compiler.compile_builtin_json_grammar();
    let time_end = Instant::now();
    println!("Time to get compiled grammar after clear: {:?}", time_end.duration_since(time_start));
    let matcher_after_clear =
        GrammarMatcher::with(&compiled_grammar_after_clear, None, Some(true), None);
    check_matcher(matcher_after_clear);
}

#[test]
fn test_grammar_compiler_json_1_thread() {
    do_test_grammar_compiler_json_test(1);
}

#[test]
fn test_grammar_compiler_json_4_thread() {
    do_test_grammar_compiler_json_test(4);
}

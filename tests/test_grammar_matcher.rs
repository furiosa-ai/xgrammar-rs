mod common;

use std::time::Instant;

use ndarray::{ArrayD, IxDyn};
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
    let compiler = GrammarCompiler::new(&tokenizer_info);

    let time_start = Instant::now();
    let compiled_grammar = compiler.compile_grammar(&grammar).expect("Failed to compile grammar");
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
    let grammar_compiler = GrammarCompiler::with(&tokenizer_info, Some(max_threads), None, None);
    let time_end = Instant::now();
    println!("Time to init cached grammar compiler: {:?}", time_end.duration_since(time_start));

    let check_matcher = |mut matcher: GrammarMatcher| {
        assert!(!matcher.is_terminated());
        assert!(!matcher.accept_string("{ name: \"John\" }", None));
        assert!(matcher.accept_string("{\"name\": \"John\"}", None));
        assert!(matcher.is_terminated());
    };

    let time_start = Instant::now();
    let compiled_grammar = grammar_compiler
        .compile_builtin_json_grammar()
        .expect("Failed to compile builtin JSON grammar");
    let time_end = Instant::now();
    println!("Time to get compiled grammar: {:?}", time_end.duration_since(time_start));
    let matcher = GrammarMatcher::with(&compiled_grammar, None, Some(true), None);
    check_matcher(matcher);

    let time_start = Instant::now();
    let compiled_grammar_again = grammar_compiler
        .compile_builtin_json_grammar()
        .expect("Failed to compile builtin JSON grammar");
    let time_end = Instant::now();
    println!("Time to get compiled grammar again: {:?}", time_end.duration_since(time_start));
    let matcher_again = GrammarMatcher::with(&compiled_grammar_again, None, Some(true), None);
    check_matcher(matcher_again);

    grammar_compiler.clear_cache();

    let time_start = Instant::now();
    let compiled_grammar_after_clear = grammar_compiler
        .compile_builtin_json_grammar()
        .expect("Failed to compile builtin JSON grammar");
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

#[test]
fn test_matcher_accept_string_and_bitmask() {
    use dlpark::prelude::*;

    // 1. Setup: Compile a simple JSON grammar
    let tokenizer_info =
        TokenizerInfo::from_pretrained(GPT_OSS_20B_PRETRAINED_ID, None, None, None)
            .expect("Failed to load tokenizer info");
    let compiler = GrammarCompiler::new(&tokenizer_info);
    let compiled_grammar =
        compiler.compile_builtin_json_grammar().expect("Failed to compile builtin JSON grammar");
    let mut matcher = GrammarMatcher::with(&compiled_grammar, None, Some(true), None);

    // 2. Initial state check
    assert!(!matcher.is_terminated(), "Matcher should not be terminated initially");

    // 3. Accept the first part of a valid string
    let partial_input = "{\"key\":";
    assert!(
        matcher.accept_string(partial_input, None),
        "Matcher should accept a valid partial string"
    );
    assert!(!matcher.is_terminated(), "Matcher should not be terminated after partial input");

    // 4. Validate fill_next_token_bitmask
    let vocab_size = tokenizer_info.get_vocab_size() as usize;
    let bitmask_len = vocab_size.div_ceil(32);
    let bitmask = ArrayD::from_shape_vec(IxDyn(&[1, bitmask_len]), vec![0i32; bitmask_len])
        .expect("fail to create a bitmask");

    // Create a DLTensor wrapping the bitmask data
    // let mut dl_tensor = SafeManagedTensorVersioned::new(vec![0i32; bitmask_len]).unwrap();
    let mut dl_tensor = SafeManagedTensorVersioned::new(bitmask).unwrap();

    // The bitmask should be modified
    let needs_application = matcher.fill_next_token_bitmask(&mut dl_tensor, None, None);
    assert!(needs_application, "Bitmask should need application for a partial match");

    // Verify that the bitmask is no longer all zeros.
    // We expect that some tokens (like quotes for a string value) are allowed.

    let bitmask_after_fill: &[i32] = dl_tensor.as_slice_contiguous().expect("fail to get slice");
    assert_eq!(bitmask_after_fill.len(), bitmask_len, "Bitmask length mismatch");
    assert!(bitmask_after_fill.iter().any(|&x| x != 0), "Bitmask should have some bits set to 1");

    // 5. Accept the rest of the string
    let remaining_input = "\"value\"}";
    assert!(
        matcher.accept_string(remaining_input, None),
        "Matcher should accept the rest of the valid string"
    );

    // 6. Final state check
    assert!(
        matcher.is_terminated(),
        "Matcher should be terminated after accepting a full valid string"
    );

    // 7. Test behavior after termination
    assert!(
        !matcher.accept_string("a", None),
        "Matcher should not accept any more tokens after termination"
    );
}

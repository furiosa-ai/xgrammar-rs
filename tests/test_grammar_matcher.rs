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
    let needs_application = matcher
        .fill_next_token_bitmask(&mut dl_tensor, None, None)
        .expect("fill_next_token_bitmask should succeed");
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

/// Test rollback with valid and invalid cases
#[test]
fn test_matcher_rollback() {
    use xgrammar::XGrammarErr;

    // Setup: Compile a simple JSON grammar
    let tokenizer_info =
        TokenizerInfo::from_pretrained(GPT_OSS_20B_PRETRAINED_ID, None, None, None)
            .expect("Failed to load tokenizer info");
    let compiler = GrammarCompiler::new(&tokenizer_info);
    let compiled_grammar =
        compiler.compile_builtin_json_grammar().expect("Failed to compile builtin JSON grammar");

    // Set max_rollback_tokens to 3 for testing
    let mut matcher = GrammarMatcher::with(&compiled_grammar, None, Some(true), Some(3));

    // Initial state: no tokens accepted yet
    assert!(!matcher.is_terminated(), "Matcher should not be terminated initially");

    // Test 1: Try to rollback when no tokens have been accepted
    let result = matcher.rollback(Some(1));
    assert!(result.is_err(), "Should fail to rollback when no tokens have been accepted");
    if let Err(XGrammarErr::MatcherError(err_msg)) = result {
        assert!(err_msg.contains(
            "Intended to rollback 1 tokens, but only the last 0 steps of history are saved"
        ));
    } else {
        panic!("Expected MatcherError");
    }

    // Test 2: Accept some tokens
    assert!(matcher.accept_string("{\"key\":", None), "Should accept partial JSON");
    assert!(matcher.accept_string("\"value\"", None), "Should accept string value");
    assert!(matcher.accept_string("}", None), "Should accept closing brace");
    assert!(matcher.is_terminated(), "Matcher should be terminated after complete JSON");

    // Test 3: Rollback 1 token (valid)
    let result = matcher.rollback(Some(1));
    assert!(result.is_ok(), "Should successfully rollback 1 token");
    assert!(!matcher.is_terminated(), "Matcher should not be terminated after rollback");

    // Test 4: Rollback 2 more tokens (valid)
    let result = matcher.rollback(Some(2));
    assert!(result.is_ok(), "Should successfully rollback 2 tokens");

    // Test 5: Try to rollback more tokens than available
    // We've accepted 3 tokens and rolled back 3, so no history left
    let result = matcher.rollback(Some(1));
    assert!(result.is_err(), "Should fail to rollback when no history is left");
    if let Err(XGrammarErr::MatcherError(err_msg)) = result {
        assert!(
            err_msg.contains("Intended to rollback") && err_msg.contains("but only the last"),
            "Error message should indicate rollback out of range, got: {}",
            err_msg
        );
    } else {
        panic!("Expected MatcherError");
    }
}

/// Test fill_next_token_bitmask error handling
#[test]
fn test_fill_next_token_bitmask_error() {
    use dlpark::prelude::*;
    use xgrammar::XGrammarErr;

    let tokenizer_info =
        TokenizerInfo::from_pretrained(GPT_OSS_20B_PRETRAINED_ID, None, None, None)
            .expect("Failed to load tokenizer info");
    let compiler = GrammarCompiler::new(&tokenizer_info);
    let compiled_grammar =
        compiler.compile_builtin_json_grammar().expect("Failed to compile builtin JSON grammar");

    let mut matcher = GrammarMatcher::with(&compiled_grammar, None, Some(false), None);

    // Accept a complete valid JSON string
    assert!(matcher.accept_string("{\"key\":\"value\"}", None), "Should accept valid JSON");

    // Test: Try to fill bitmask with invalid parameters (wrong dtype/shape)
    let wrong_bitmask_len = 10; // Too small
    let bitmask =
        ArrayD::from_shape_vec(IxDyn(&[1, wrong_bitmask_len]), vec![0i32; wrong_bitmask_len])
            .expect("fail to create a bitmask");
    let mut dl_tensor = SafeManagedTensorVersioned::new(bitmask).unwrap();

    let result = matcher.fill_next_token_bitmask(&mut dl_tensor, None, None);
    let Err(XGrammarErr::MatcherError(err_msg)) = result else {
        panic!("Expected MatcherError");
    };

    assert!(
        err_msg.contains("The provided bitmask's shape is not valid: should be (batch_size, 6251)")
    );
}

#[test]
fn test_stop_token_early_termination() {
    use xgrammar::GrammarCompiler;

    let tokenizer =
        common::load_tokenizer(GPT_OSS_20B_PRETRAINED_ID).expect("Failed to load tokenizer");

    let encoding = tokenizer.encode(".", false).expect("Failed to encode '.'");
    let period_token_ids: Vec<i32> = encoding.get_ids().iter().map(|&id| id as i32).collect();

    assert!(!period_token_ids.is_empty(), "Should find at least one token for '.'");
    let period_token_id = period_token_ids[0];

    let regex_pattern = r"[a-zA-Z0-9 ,]+\.?";

    let tokenizer_info_without_stop =
        TokenizerInfo::from_pretrained(GPT_OSS_20B_PRETRAINED_ID, None, None, None)
            .expect("Failed to load tokenizer info");

    let compiler = GrammarCompiler::new(&tokenizer_info_without_stop);
    let compiled_grammar = compiler.compile_regex(regex_pattern).expect("Failed to compile regex");

    let mut matcher_without_stop = GrammarMatcher::with(&compiled_grammar, None, Some(true), None);

    let test_text = "Hello world";
    let text_encoding = tokenizer.encode(test_text, false).expect("Failed to encode test text");

    for &token_id in text_encoding.get_ids() {
        let accepted = matcher_without_stop.accept_token(token_id as i32, None);
        assert!(accepted, "Token should be accepted");
    }

    let terminated_before_period = matcher_without_stop.is_terminated();
    let period_accepted = matcher_without_stop.accept_token(period_token_id, None);
    let terminated_after_period = matcher_without_stop.is_terminated();

    let tokenizer_info_with_stop = TokenizerInfo::from_pretrained(
        GPT_OSS_20B_PRETRAINED_ID,
        None,
        None,
        Some(period_token_ids.clone()),
    )
    .expect("Failed to load tokenizer info with stop tokens");

    let compiler_with_stop = GrammarCompiler::new(&tokenizer_info_with_stop);
    let compiled_grammar_with_stop = compiler_with_stop
        .compile_regex(regex_pattern)
        .expect("Failed to compile regex with stop tokens");

    let mut matcher_with_stop =
        GrammarMatcher::with(&compiled_grammar_with_stop, None, Some(false), None);

    for &token_id in text_encoding.get_ids() {
        let accepted = matcher_with_stop.accept_token(token_id as i32, None);
        assert!(accepted, "Token should be accepted");
    }

    let terminated_before_stop_token = matcher_with_stop.is_terminated();
    let period_accepted_with_stop = matcher_with_stop.accept_token(period_token_id, None);
    let terminated_after_stop_token = matcher_with_stop.is_terminated();

    assert!(
        terminated_before_period,
        "Without stop token, matcher should terminate when grammar is complete"
    );
    assert!(period_accepted, "Period token should be accepted");
    assert!(terminated_after_period, "Matcher should still be terminated after accepting period");

    assert!(
        !terminated_before_stop_token,
        "With stop token and terminate_without_stop_token=false, matcher should not terminate before stop token"
    );

    if period_accepted_with_stop {
        assert!(
            terminated_after_stop_token,
            "Matcher should be terminated immediately after accepting stop token"
        );
    }
}

/// Test that `is_completed` returns true once the root rule is fully matched,
/// even though the stop token has not been accepted yet.
#[test]
fn test_matcher_is_completed_without_stop_token() {
    let tokenizer_info =
        TokenizerInfo::from_pretrained(GPT_OSS_20B_PRETRAINED_ID, None, None, None)
            .expect("Failed to load tokenizer info");
    let compiler = GrammarCompiler::new(&tokenizer_info);
    let compiled_grammar =
        compiler.compile_builtin_json_grammar().expect("Failed to compile builtin JSON grammar");
    // terminate_without_stop_token = false: matcher does NOT auto-terminate when root rule matches.
    let mut matcher = GrammarMatcher::with(&compiled_grammar, None, Some(false), None);

    assert!(!matcher.is_completed());
    assert!(!matcher.is_terminated());

    assert!(matcher.accept_string("{\"name\":\"John\"}", None));
    assert!(matcher.is_completed(), "grammar root rule fully matched");
    assert!(!matcher.is_terminated(), "stop token not yet accepted");
}

/// Test that `fork` creates an independent deep copy of the matcher state.
#[test]
fn test_matcher_fork_is_deep_copy() {
    let tokenizer_info =
        TokenizerInfo::from_pretrained(GPT_OSS_20B_PRETRAINED_ID, None, None, None)
            .expect("Failed to load tokenizer info");
    let compiler = GrammarCompiler::new(&tokenizer_info);
    let compiled_grammar =
        compiler.compile_builtin_json_grammar().expect("Failed to compile builtin JSON grammar");
    let mut matcher = GrammarMatcher::with(&compiled_grammar, None, Some(true), None);

    // Drive to an open JSON object.
    assert!(matcher.accept_string("{\"a\":", None));

    // Fork — both branches should be independently valid.
    let mut forked = matcher.fork();
    assert!(matcher.accept_string("1}", None));
    assert!(matcher.is_terminated());

    assert!(!forked.is_terminated(), "forked matcher should retain pre-fork state");
    assert!(forked.accept_string("\"x\"}", None));
    assert!(forked.is_terminated());
}

/// Regression test for `GrammarMatcher::get_stop_token_ids` and `TokenizerInfo::get_decoded_vocab`
/// — both previously relied on Rust `Vec<T>` / C++ `std::vector<T>` layout compatibility,
/// which is NOT guaranteed on libstdc++. They now use an explicit size+data-pointer
/// marshalling path; this test ensures the result is sane.
#[test]
fn test_vector_returning_getters_do_not_crash() {
    let tokenizer_info =
        TokenizerInfo::from_pretrained(GPT_OSS_20B_PRETRAINED_ID, None, None, None)
            .expect("Failed to load tokenizer info");
    let vocab_size = tokenizer_info.get_vocab_size();
    let decoded_vocab = tokenizer_info.get_decoded_vocab();
    assert_eq!(decoded_vocab.len(), vocab_size as usize);

    let compiler = GrammarCompiler::new(&tokenizer_info);
    let compiled_grammar =
        compiler.compile_builtin_json_grammar().expect("Failed to compile builtin JSON grammar");
    let matcher = GrammarMatcher::with(&compiled_grammar, None, Some(true), None);
    let stop_ids = matcher.get_stop_token_ids();
    // We don't know the exact set, but it should be a small non-empty list and every id should
    // fit in the tokenizer vocab range.
    assert!(!stop_ids.is_empty(), "expected at least one stop token");
    for id in &stop_ids {
        assert!(*id >= 0 && *id < vocab_size, "stop id {id} out of vocab range");
    }
}

/// Test GrammarMatcher with JSON schema that has both properties and patternProperties.
//
// cf. https://github.com/mlc-ai/xgrammar/pull/594 , which was resolved in xgrammar v0.2.1.
#[test]
fn test_grammar_from_json_schema_with_pattern_properties() {
    let json_schema = serde_json::json!({
      "$schema": "http://json-schema.org/draft-04/schema#",
      "patternProperties": {
        "^(cat|dog)_name$": { "type": "string" },
        "^(extra_field_[0-9]+)$": { "type": ["string", "integer", "null"] }
      },
      "properties": { "name": { "type": "string" } },
      "additionalProperties": false,
      "required": ["name"],
      "type": "object"
    })
    .to_string();
    let grammar =
        Grammar::from_json_schema(&json_schema, None, None, None, None, None, None).unwrap();
    let tokenizer_info =
        TokenizerInfo::from_pretrained(GPT_OSS_20B_PRETRAINED_ID, None, None, None)
            .expect("Failed to load tokenizer info");
    let compiler = GrammarCompiler::new(&tokenizer_info);

    let compiled_grammar = compiler.compile_grammar(&grammar).expect("should be a valid grammar");
    let mut matcher = GrammarMatcher::with(&compiled_grammar, None, Some(true), None);

    // NOTE: we use raw strings here instead of `serde_json::Value::to_string()`,
    // because the default compilation parameters require ordering between properties (required ones come first).
    for sample_json in [
        r#"{"name": ""}"#,
        r#"{"name": "john"}"#,
        r#"{"name": "john", "cat_name": "mocha"}"#,
        r#"{"name": "john", "dog_name": "joy"}"#,
        r#"{"name": "john", "extra_field_123": 123}"#,
        r#"{"name": "john", "extra_field_1": "test"}"#,
        r#"{"name": "john", "extra_field_1": "one", "extra_field_2": 2, "extra_field_3": null}"#,
        r#"{"name": "john", "cat_name": "mocha", "dog_name": "joy", "extra_field_1": "hello"}"#,
    ] {
        matcher.reset();
        assert!(matcher.accept_string(sample_json, None), "should accept {}", sample_json);
        assert!(matcher.is_terminated());
    }

    for sample_json in [
        r#"{"name": "evil", "cat_name": 3}"#,
        r#"{"name": "evil", "unexpected_field": "alien"}"#,
        r#"{"name": "evil", "extra_field_": "no number"}"#,
    ] {
        matcher.reset();
        assert!(!matcher.accept_string(sample_json, None), "should not accept {}", sample_json);
    }
}

// ---------------------------------------------------------------------------
// traverse_draft_tree (speculative decoding)
// ---------------------------------------------------------------------------

type Tensor = dlpark::versioned::SafeManagedTensorVersioned;

/// Encode `s` and assert it maps to exactly one token, returning the id as
/// `i64` for draft-tree tensors. The traverse tests depend on these strings
/// being single tokens; this fails loudly if a tokenizer revision changes that.
fn single_token_id(tokenizer: &xgrammar::tokenizers::Tokenizer, s: &str) -> i64 {
    let encoding = tokenizer.encode(s, false).expect("failed to encode");
    let ids = encoding.get_ids();
    assert_eq!(ids.len(), 1, "expected {s:?} to encode to exactly one token, got {ids:?}");
    ids[0] as i64
}

fn i64_tensor(values: Vec<i64>) -> Tensor {
    Tensor::new(values).expect("failed to create i64 tensor")
}

/// Allocate a `(rows, len)` int32 bitmask tensor pre-filled with `fill`.
/// Traverse tests pre-fill with -1 so a row zeroed by the traversal is
/// distinguishable from a row it never touched.
fn new_bitmask_filled(rows: usize, len: usize, fill: i32) -> Tensor {
    let arr = ArrayD::from_elem(IxDyn(&[rows, len]), fill);
    Tensor::new(arr).expect("failed to create bitmask tensor")
}

fn bitmask_rows(tensor: &Tensor, len: usize) -> Vec<Vec<i32>> {
    use dlpark::prelude::*;
    let slice: &[i32] = tensor.as_slice_contiguous().expect("failed to get bitmask slice");
    slice.chunks(len).map(|chunk| chunk.to_vec()).collect()
}

/// Returns the tokenizer (for deriving draft token ids), the compiled builtin
/// JSON grammar, and the per-row bitmask length for its vocabulary.
fn setup_traverse_fixture() -> (xgrammar::tokenizers::Tokenizer, xgrammar::CompiledGrammar, usize) {
    let tokenizer =
        common::load_tokenizer(GPT_OSS_20B_PRETRAINED_ID).expect("Failed to load tokenizer");
    let tokenizer_info =
        TokenizerInfo::from_pretrained(GPT_OSS_20B_PRETRAINED_ID, None, None, None)
            .expect("Failed to load tokenizer info");
    let bitmask_len = xgrammar::get_bitmask_size(tokenizer_info.get_vocab_size()) as usize;
    let compiler = GrammarCompiler::new(&tokenizer_info);
    let compiled =
        compiler.compile_builtin_json_grammar().expect("Failed to compile builtin JSON grammar");
    (tokenizer, compiled, bitmask_len)
}

#[test]
fn test_get_bitmask_size() {
    use xgrammar::get_bitmask_size;

    assert_eq!(get_bitmask_size(1), 1);
    assert_eq!(get_bitmask_size(32), 1);
    assert_eq!(get_bitmask_size(33), 2);
    // A realistic vocabulary size (gpt-oss-20b).
    assert_eq!(get_bitmask_size(201_088), 201_088usize.div_ceil(32) as i32);
}

#[test]
fn test_traverse_draft_tree_linear() {
    use dlpark::prelude::*;

    let (tokenizer, compiled, bitmask_len) = setup_traverse_fixture();
    let mut matcher = GrammarMatcher::new(&compiled);

    let brace = single_token_id(&tokenizer, "{");
    let quote = single_token_id(&tokenizer, "\"");

    // Linear tree 0 -> 1 -> 2; draft_tokens[0] is ignored by the traversal.
    let next_token = i64_tensor(vec![1, 2, -1]);
    let next_sibling = i64_tensor(vec![-1, -1, -1]);
    let draft_tokens = i64_tensor(vec![0, brace, quote]);
    let mut bitmask = new_bitmask_filled(3, bitmask_len, -1);

    let completed = matcher
        .traverse_draft_tree(&next_token, &next_sibling, &draft_tokens, &mut bitmask, None)
        .expect("traverse_draft_tree should succeed");
    assert!(completed);

    let rows = bitmask_rows(&bitmask, bitmask_len);
    for (i, row) in rows.iter().enumerate() {
        assert!(row.iter().any(|&w| w != -1), "row {i} was never written");
        assert!(row.iter().any(|&w| w != 0), "row {i} should have allowed tokens");
    }

    // Row 0 must equal fill_next_token_bitmask of a fresh matcher: both
    // reflect the same (initial) matcher state.
    let mut fresh = GrammarMatcher::new(&compiled);
    let mut root_only = new_bitmask_filled(1, bitmask_len, 0);
    fresh.fill_next_token_bitmask(&mut root_only, None, None).expect("fill should succeed");
    let root_row: &[i32] = root_only.as_slice_contiguous().expect("failed to get slice");
    assert_eq!(rows[0], root_row);

    // A completed traversal leaves the matcher state untouched.
    assert!(matcher.accept_string("{\"a\":1}", None));
    assert!(matcher.is_terminated());
}

#[test]
fn test_traverse_draft_tree_with_siblings() {
    let (tokenizer, compiled, bitmask_len) = setup_traverse_fixture();
    let mut matcher = GrammarMatcher::new(&compiled);

    let brace = single_token_id(&tokenizer, "{");
    let bracket = single_token_id(&tokenizer, "[");

    // Root with two children: nodes 1 and 2 are siblings, both grammar-valid
    // continuations. Exercises the internal rollback between siblings.
    let next_token = i64_tensor(vec![1, -1, -1]);
    let next_sibling = i64_tensor(vec![-1, 2, -1]);
    let draft_tokens = i64_tensor(vec![0, brace, bracket]);
    let mut bitmask = new_bitmask_filled(3, bitmask_len, -1);

    let completed = matcher
        .traverse_draft_tree(&next_token, &next_sibling, &draft_tokens, &mut bitmask, None)
        .expect("traverse_draft_tree should succeed");
    assert!(completed);

    let rows = bitmask_rows(&bitmask, bitmask_len);
    for (i, row) in rows.iter().enumerate() {
        assert!(row.iter().any(|&w| w != -1), "row {i} was never written");
        assert!(row.iter().any(|&w| w != 0), "row {i} should have allowed tokens");
    }
}

#[test]
fn test_traverse_draft_tree_rejected_node() {
    let (tokenizer, compiled, bitmask_len) = setup_traverse_fixture();
    let mut matcher = GrammarMatcher::new(&compiled);

    let brace = single_token_id(&tokenizer, "{");
    let colon = single_token_id(&tokenizer, ":");

    // ":" is not a valid JSON continuation right after "{": node 2 is
    // rejected, its row is zeroed, and the traversal still completes.
    let next_token = i64_tensor(vec![1, 2, -1]);
    let next_sibling = i64_tensor(vec![-1, -1, -1]);
    let draft_tokens = i64_tensor(vec![0, brace, colon]);
    let mut bitmask = new_bitmask_filled(3, bitmask_len, -1);

    let completed = matcher
        .traverse_draft_tree(&next_token, &next_sibling, &draft_tokens, &mut bitmask, None)
        .expect("traverse_draft_tree should succeed");
    assert!(completed, "a rejected draft token must not fail the traversal");

    let rows = bitmask_rows(&bitmask, bitmask_len);
    assert!(rows[1].iter().any(|&w| w != 0), "row 1 should have allowed tokens");
    assert!(rows[2].iter().all(|&w| w == 0), "rejected node's row must be zeroed");
}

#[test]
fn test_traverse_draft_tree_skipped_subtree() {
    let (tokenizer, compiled, bitmask_len) = setup_traverse_fixture();
    let mut matcher = GrammarMatcher::new(&compiled);

    let brace = single_token_id(&tokenizer, "{");

    // Node 1 carries an out-of-range token id: its row is zeroed and its
    // subtree (node 2) is skipped without its row being touched.
    let next_token = i64_tensor(vec![1, 2, -1]);
    let next_sibling = i64_tensor(vec![-1, -1, -1]);
    let draft_tokens = i64_tensor(vec![0, 10_000_000, brace]);
    let mut bitmask = new_bitmask_filled(3, bitmask_len, -1);

    let completed = matcher
        .traverse_draft_tree(&next_token, &next_sibling, &draft_tokens, &mut bitmask, None)
        .expect("traverse_draft_tree should succeed");
    assert!(completed);

    let rows = bitmask_rows(&bitmask, bitmask_len);
    assert!(rows[1].iter().all(|&w| w == 0), "out-of-range node's row must be zeroed");
    assert!(rows[2].iter().all(|&w| w == -1), "skipped subtree's row must remain untouched");
}

#[test]
fn test_traverse_draft_tree_validation_errors() {
    use xgrammar::XGrammarErr;

    let (_tokenizer, compiled, bitmask_len) = setup_traverse_fixture();
    let mut matcher = GrammarMatcher::new(&compiled);

    fn expect_matcher_err(result: xgrammar::Result<bool>, needle: &str) {
        let Err(XGrammarErr::MatcherError(err_msg)) = result else {
            panic!("Expected MatcherError containing {needle:?}");
        };
        assert!(err_msg.contains(needle), "expected {needle:?} in error message: {err_msg}");
    }

    let next_token = i64_tensor(vec![1, 2, -1]);
    let next_sibling = i64_tensor(vec![-1, -1, -1]);
    let draft_tokens = i64_tensor(vec![0, 1, 2]);
    // One bitmask serves every case: validation fails before any row is written.
    let mut bitmask = new_bitmask_filled(3, bitmask_len, 0);

    // int32 draft tokens instead of int64.
    let draft_i32 = Tensor::new(vec![0i32, 1, 2]).unwrap();
    expect_matcher_err(
        matcher.traverse_draft_tree(&next_token, &next_sibling, &draft_i32, &mut bitmask, None),
        "The draft_tokens tensor must be a 1D int64 tensor",
    );

    // Length mismatch between the tree tensors.
    let short_sibling = i64_tensor(vec![-1, -1]);
    expect_matcher_err(
        matcher.traverse_draft_tree(&next_token, &short_sibling, &draft_tokens, &mut bitmask, None),
        "must have the same length",
    );

    // 1-D bitmask instead of 2-D.
    let mut flat_bitmask = Tensor::new(vec![0i32; 3 * bitmask_len]).unwrap();
    expect_matcher_err(
        matcher.traverse_draft_tree(
            &next_token,
            &next_sibling,
            &draft_tokens,
            &mut flat_bitmask,
            None,
        ),
        "The token_bitmask tensor must be a 2D int32 tensor",
    );

    // Bitmask row count not matching the node count.
    let mut short_bitmask = new_bitmask_filled(2, bitmask_len, 0);
    expect_matcher_err(
        matcher.traverse_draft_tree(
            &next_token,
            &next_sibling,
            &draft_tokens,
            &mut short_bitmask,
            None,
        ),
        "The token_bitmask batch size must match the number of nodes",
    );

    // The root node must not have a sibling.
    let bad_sibling = i64_tensor(vec![1, -1, -1]);
    expect_matcher_err(
        matcher.traverse_draft_tree(&next_token, &bad_sibling, &draft_tokens, &mut bitmask, None),
        "The root node must not have siblings",
    );
}

#[test]
fn test_traverse_draft_tree_timeout() {
    let (tokenizer, compiled, bitmask_len) = setup_traverse_fixture();
    let mut matcher = GrammarMatcher::new(&compiled);

    // Drive the matcher to a state where a number value starts: the builtin
    // JSON grammar's root only allows "{" or "[", so a bare digit would be
    // rejected at the root state and the chain below would be skipped without
    // ever reaching the timeout check.
    assert!(matcher.accept_string("{\"a\":", None));

    // A long linear chain of "1" tokens: each node extends the JSON number, so
    // every node is grammar-valid and only the timeout can stop the traversal.
    let one = single_token_id(&tokenizer, "1");
    const NUM_NODES: usize = 1000;
    let mut next_token: Vec<i64> = (1..NUM_NODES as i64).collect();
    next_token.push(-1);

    let next_token = i64_tensor(next_token);
    let next_sibling = i64_tensor(vec![-1; NUM_NODES]);
    let draft_tokens = i64_tensor(vec![one; NUM_NODES]);
    let mut bitmask = new_bitmask_filled(NUM_NODES, bitmask_len, -1);

    let completed = matcher
        .traverse_draft_tree(&next_token, &next_sibling, &draft_tokens, &mut bitmask, Some(1e-7))
        .expect("traverse_draft_tree should succeed");
    assert!(!completed, "a 1e-7s threshold must time out on a 1000-node chain");

    // The root row is always filled: the timeout is only checked at non-root
    // nodes. Read it straight off the contiguous slice — copying all 1000 rows
    // via bitmask_rows just to inspect row 0 would be wasteful.
    use dlpark::prelude::*;
    let slice: &[i32] = bitmask.as_slice_contiguous().expect("failed to get bitmask slice");
    let root_row = &slice[..bitmask_len];
    assert!(root_row.iter().any(|&w| w != -1), "root row must be written even on timeout");
    assert!(root_row.iter().any(|&w| w != 0), "root row should have allowed tokens");

    // After a timeout the matcher state is not guaranteed to be restored;
    // reset() returns it to a usable state (the documented recovery path).
    matcher.reset();
    assert!(matcher.accept_string("{\"a\":1}", None));
}

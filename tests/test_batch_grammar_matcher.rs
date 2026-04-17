mod common;

use dlpark::prelude::*;
use ndarray::{ArrayD, IxDyn};
use xgrammar::{BatchGrammarMatcher, GrammarCompiler, GrammarMatcher, TokenizerInfo};

const GPT_OSS_20B_PRETRAINED_ID: &str = "openai/gpt-oss-20b";

fn setup_matchers(count: usize) -> (TokenizerInfo, Vec<GrammarMatcher>) {
    let tokenizer_info =
        TokenizerInfo::from_pretrained(GPT_OSS_20B_PRETRAINED_ID, None, None, None)
            .expect("Failed to load tokenizer info");
    let compiler = GrammarCompiler::new(&tokenizer_info);
    let compiled_grammar =
        compiler.compile_builtin_json_grammar().expect("Failed to compile builtin JSON grammar");
    let matchers = (0..count)
        .map(|_| GrammarMatcher::with(&compiled_grammar, None, Some(true), None))
        .collect();
    (tokenizer_info, matchers)
}

#[test]
fn test_batch_accept_string() {
    let (_tokenizer_info, mut matchers) = setup_matchers(3);
    let inputs = ["{\"a\":1}", "{ a: 1 }", "{\"b\":\"c\"}"]; // valid, invalid, valid
    let results = BatchGrammarMatcher::batch_accept_string(&mut matchers, &inputs, None);
    assert_eq!(results, vec![true, false, true]);
    assert!(matchers[0].is_terminated());
    assert!(!matchers[1].is_terminated());
    assert!(matchers[2].is_terminated());
}

#[test]
fn test_batch_accept_token_matches_single_matcher() {
    // batch_accept_token should produce the same result as per-matcher accept_token.
    let (_tokenizer_info, mut matchers) = setup_matchers(2);
    let (_ti2, mut reference) = setup_matchers(2);

    // Pick a token id that is a valid JSON starter ("{" is ASCII 0x7B = 123 — but token ids are
    // model-specific). Fall back to a universally-accepted '{' via accept_string first, then
    // exercise batch_accept_token on a known token that all matchers reject so results compare.
    //
    // For correctness, use a token id that is out of the tokenizer's vocab range → both paths
    // should reject it deterministically.
    let invalid_token: i32 = -1;
    let batch_results = BatchGrammarMatcher::batch_accept_token(
        &mut matchers,
        &[invalid_token, invalid_token],
        None,
    );
    let ref_results: Vec<bool> =
        reference.iter_mut().map(|m| m.accept_token(invalid_token, None)).collect();
    assert_eq!(batch_results, ref_results);
}

#[test]
fn test_batch_rollback_heterogeneous() {
    let (_tokenizer_info, mut matchers) = setup_matchers(2);
    // accept_string treats the whole string as one rollback step.
    let prefill = ["{\"a\":1}", "{\"b\":2}"]; // both full valid JSONs — matchers terminate
    let accepted = BatchGrammarMatcher::batch_accept_string(&mut matchers, &prefill, None);
    assert_eq!(accepted, vec![true, true]);
    assert!(matchers[0].is_terminated());
    assert!(matchers[1].is_terminated());

    // Roll back matcher 0 by 1 step (undo the whole prefill), matcher 1 by 0 steps (no-op).
    BatchGrammarMatcher::batch_rollback(&mut matchers, &[1, 0]);

    // Matcher 0 is back to the initial state and should accept a fresh JSON.
    assert!(!matchers[0].is_terminated());
    assert!(matchers[0].accept_string("{\"x\":1}", None));
    assert!(matchers[0].is_terminated());

    // Matcher 1 remains terminated (rollback of 0).
    assert!(matchers[1].is_terminated());
}

#[test]
fn test_batch_fill_next_token_bitmask() {
    let (tokenizer_info, mut matchers) = setup_matchers(3);
    // Drive each matcher to a partial state so the bitmask is non-trivial.
    let prefills = ["{\"a\":", "{\"b\":", "{\"c\":"];
    BatchGrammarMatcher::batch_accept_string(&mut matchers, &prefills, None);

    let vocab_size = tokenizer_info.get_vocab_size() as usize;
    let bitmask_len = vocab_size.div_ceil(32);
    let shape = [matchers.len(), bitmask_len];
    let bitmask = ArrayD::from_shape_vec(IxDyn(&shape), vec![0i32; matchers.len() * bitmask_len])
        .expect("fail to create bitmask");
    let mut dl_tensor = SafeManagedTensorVersioned::new(bitmask).unwrap();

    let mut batch = BatchGrammarMatcher::new();
    batch
        .batch_fill_next_token_bitmask(&mut matchers, &mut dl_tensor, None, None)
        .expect("batch_fill_next_token_bitmask should succeed");

    let slice: &[i32] = dl_tensor.as_slice_contiguous().expect("fail to get slice");
    assert_eq!(slice.len(), matchers.len() * bitmask_len);
    assert!(slice.iter().any(|&x| x != 0), "Bitmask should have some bits set");
}

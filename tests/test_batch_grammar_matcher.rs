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
    // batch_accept_token should produce the same per-matcher results as
    // calling accept_token on each matcher individually — for any input, at
    // any matcher state. We drive the batch and reference matchers through
    // identical prefills, then issue the same batch of token ids and compare.
    let (_tokenizer_info, mut matchers) = setup_matchers(3);
    let (_ti2, mut reference) = setup_matchers(3);

    // Prefill each matcher pair to a distinct state.
    // matcher 0: fresh (empty)
    // matcher 1: mid-JSON (`{"a":`)
    // matcher 2: fully matched (`{"a":1}`, terminated)
    for m in [&mut matchers, &mut reference] {
        assert!(m[1].accept_string("{\"a\":", None));
        assert!(m[2].accept_string("{\"a\":1}", None));
        assert!(m[2].is_terminated());
    }

    // Exercise the batch with an out-of-range token id — every matcher must
    // reject it deterministically — and compare element-by-element against the
    // single-matcher path.
    let invalid_token: i32 = -1;
    let token_ids = vec![invalid_token; matchers.len()];
    let batch_results = BatchGrammarMatcher::batch_accept_token(&mut matchers, &token_ids, None);
    let ref_results: Vec<bool> = reference
        .iter_mut()
        .zip(token_ids.iter())
        .map(|(m, &id)| m.accept_token(id, None))
        .collect();
    assert_eq!(batch_results, ref_results, "batch and per-matcher paths must agree");

    // Post-call state must match as well (batch must not have silently advanced
    // or rolled back any matcher beyond what the single-matcher path did).
    for (b, r) in matchers.iter().zip(reference.iter()) {
        assert_eq!(b.is_terminated(), r.is_terminated());
    }
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

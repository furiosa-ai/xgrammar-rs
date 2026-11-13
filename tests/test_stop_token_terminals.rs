use dlpark::{SafeManagedTensorVersioned, traits::TensorView};
use ndarray::{ArrayD, IxDyn};
use serde_json::json;
use xgrammar::{GrammarCompiler, GrammarMatcher, TokenizerInfo};

const RETURN: i32 = 4;
const CALL: i32 = 5;
const VOCAB_SIZE: usize = 6;

fn allowed_tokens(matcher: &mut GrammarMatcher) -> Vec<bool> {
    let mut mask = SafeManagedTensorVersioned::new(ArrayD::<i32>::zeros(IxDyn(&[1, 1]))).unwrap();
    matcher.fill_next_token_bitmask(&mut mask, None, None).unwrap();
    let words: &[i32] = mask.as_slice_contiguous().unwrap();
    (0..VOCAB_SIZE).map(|id| words[id / 32] & (1 << (id % 32)) != 0).collect()
}

/// A stop token may also close a structural tag. Check mask/acceptance agreement,
/// rejection without state changes, termination, rollback, and reset, without Hub assets.
/// Reproduces https://github.com/mlc-ai/xgrammar/issues/701 for default and overridden stops.
#[test]
fn test_stop_tokens_that_complete_structural_tags() {
    let tokenizer = json!({
        "model": {"type": "WordLevel", "unk_token": "ok", "vocab": {
            "<header>": 0, "ok": 1, "{\"bar\":": 2, "\"ok\"}": 3,
            "<|return|>": RETURN, "<|call|>": CALL
        }}
    });
    let info = TokenizerInfo::from_backend_str(&tokenizer.to_string(), None, vec![RETURN]).unwrap();
    let compiler = GrammarCompiler::new(&info);
    for (end, stop, content, body) in [
        ("<|return|>", RETURN, json!({"type": "any_text"}), vec![1]),
        (
            "<|call|>",
            CALL,
            json!({"type": "json_schema", "json_schema": {
                "type": "object", "properties": {"bar": {"type": "string"}},
                "required": ["bar"], "additionalProperties": false
            }}),
            vec![2, 3],
        ),
    ] {
        let tag = json!({"type": "structural_tag", "format": {
            "type": "tag", "begin": "<header>", "content": content, "end": end
        }});
        let compiled = compiler.compile_structural_tag(&tag.to_string()).unwrap();
        let overrides = [RETURN, CALL];
        let mut matcher = GrammarMatcher::with(
            &compiled,
            (stop == CALL).then_some(overrides.as_slice()),
            Some(false),
            None,
        );

        for _ in 0..2 {
            // An early stop must be rejected without corrupting the next mask.
            let initial = allowed_tokens(&mut matcher);
            assert!(!initial[stop as usize]);
            assert!(!matcher.accept_token(stop, None));
            assert_eq!(allowed_tokens(&mut matcher), initial);
            assert!(initial[0]);
            assert!(matcher.accept_token(0, None));

            for &token in &body {
                let allowed = allowed_tokens(&mut matcher);
                assert!(allowed[token as usize]);
                assert!(matcher.accept_token(token, None));
                assert!(!matcher.is_terminated());
                if token == 2 {
                    assert!(!allowed_tokens(&mut matcher)[CALL as usize]);
                    assert!(!matcher.accept_token(CALL, None));
                }
            }

            let before_stop = allowed_tokens(&mut matcher);
            assert!(before_stop[stop as usize]);
            assert_eq!(allowed_tokens(&mut matcher), before_stop);
            assert!(!matcher.is_completed());
            assert!(matcher.accept_token(stop, None));
            assert!(matcher.is_completed());
            assert!(matcher.is_terminated());

            matcher.rollback(Some(1)).unwrap();
            assert!(!matcher.is_terminated());
            assert_eq!(allowed_tokens(&mut matcher), before_stop);
            assert!(matcher.accept_token(stop, None));
            assert!(matcher.is_terminated());
            matcher.reset();
            assert!(!matcher.is_terminated());
            assert_eq!(allowed_tokens(&mut matcher), initial);
        }
    }
}

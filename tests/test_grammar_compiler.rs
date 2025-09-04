mod common;

use tracing::Level;
use xgrammar::{GrammarCompiler, TokenizerInfo, VocabType};

const EXAONE_4_0_32B_PRETRAINED_ID: &str = "LGAI-EXAONE/EXAONE-4.0-32B";

#[test]
#[cfg(feature = "hf_hub")]
fn test_grammar_compiler() {
    let tok_info = TokenizerInfo::from_pretrained(EXAONE_4_0_32B_PRETRAINED_ID, None, None, None)
        .expect("Failed to load tokenizer info");
    let mut compiler = GrammarCompiler::new(&tok_info);
    let compiled_grammar = compiler.compile_builtin_json_grammar();

    assert!(compiled_grammar.memory_size_bytes() > 0);
    assert_eq!(compiled_grammar.get_tokenizer_info().get_vocab_size(), 102400);
    assert_eq!(compiled_grammar.get_tokenizer_info().get_vocab_type(), VocabType::ByteLevel);
}

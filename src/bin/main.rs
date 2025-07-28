use xgrammar_rs::xgrammar::tokenizer_info::TokenizerInfo;
const EXAONE_4_0_32B_PRETRAINED_ID: &str = "LGAI-EXAONE/EXAONE-4.0-32B";

fn main() {
  use tokenizers::tokenizer::Tokenizer;

  let tokenizer = Tokenizer::from_pretrained(EXAONE_4_0_32B_PRETRAINED_ID, None)
      .expect("Failed to load tokenizer");
  TokenizerInfo::detect_metadata_from_hf(&tokenizer.to_string(false).unwrap());
}

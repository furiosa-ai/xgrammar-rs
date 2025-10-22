mod common;

use dlpark::prelude::*;
use ndarray::{Array1, ArrayD, IxDyn};
use ndarray_rand::RandomExt;
use rand::SeedableRng;
use rand::distributions::Distribution;
use rand::rngs::StdRng;
use xgrammar::{GrammarMatcher, TokenId, TokenizerInfo};

const GPT_OSS_20B_PRETRAINED_ID: &str = "openai/gpt-oss-20b";

type Logit = f32;

/// A uniform distribution for f32 values
struct F32Uniform {
    low: f32,
    high: f32,
}

impl F32Uniform {
    fn new(low: f32, high: f32) -> Self {
        Self { low, high }
    }
}

impl Distribution<Logit> for F32Uniform {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Logit {
        let uniform = rand::distributions::Uniform::new(self.low, self.high);
        rng.sample(uniform)
    }
}

/// Generate text using a grammar matcher with random logits and mask application.
/// This function generates random logits for each step, applies the grammar mask,
/// and selects the token with the highest logit value (greedy search).
///
/// # Arguments
/// * `matcher` - The grammar matcher to use
/// * `tokenizer_id` - The tokenizer ID to load from HuggingFace Hub
/// * `max_generation` - Maximum number of tokens to generate
/// * `seed` - Random seed for reproducibility
///
/// # Returns
/// * The generated text string
pub fn generate_dummy_text(
    matcher: &mut GrammarMatcher,
    tokenizer_id: &str,
    max_generation: usize,
    seed: u64,
) -> Result<String, Box<dyn std::error::Error>> {
    // Load the tokenizer using common util function
    let tokenizer = common::load_tokenizer(tokenizer_id)
        .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;

    let mut rng = StdRng::seed_from_u64(seed);
    let vocab_size = tokenizer.get_vocab_size(true);
    let bitmask_len = vocab_size.div_ceil(32);
    let mut sampled: Vec<u32> = Vec::with_capacity(max_generation);

    tracing::info!(
        "Starting generation with vocab_size={}, bitmask_len={}",
        vocab_size,
        bitmask_len
    );

    for step in 0..max_generation {
        if matcher.is_terminated() {
            tracing::info!("Matcher terminated at step {}", step);
            break;
        }

        tracing::info!("Step {}: generating token", step);

        // Generate random logits for all tokens
        let mut logits =
            Array1::<Logit>::random_using(vocab_size, F32Uniform::new(-1.0, 1.0), &mut rng);

        // Create a bitmask to get valid tokens
        let bitmask = ArrayD::from_shape_vec(IxDyn(&[1, bitmask_len]), vec![0i32; bitmask_len])?;
        let mut dl_tensor = SafeManagedTensorVersioned::new(bitmask)?;

        // Get the mask of valid tokens
        let applied = matcher.fill_next_token_bitmask(&mut dl_tensor, None, None)?;

        if applied {
            // Apply the bitmask to logits: set logit to -inf for invalid tokens (bit = 0)
            let bitmask_slice: &[i32] = dl_tensor.as_slice_contiguous()?;

            for (idx, &mask_word) in bitmask_slice.iter().enumerate() {
                for bit in 0..32 {
                    let token_id = idx * 32 + bit;
                    if token_id >= vocab_size {
                        break;
                    }

                    // If the bit is 0, set logit to -inf
                    if (mask_word & (1 << bit)) == 0 {
                        logits[token_id] = f32::NEG_INFINITY;
                    }
                }
            }
        }

        // corresponding to argmax over logits (greedy search)
        let tok_idx = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .ok_or("No valid token found")? as i32;

        tracing::info!("Step {}: selected token {}", step, tok_idx);
        sampled.push(tok_idx as u32);

        // Accept the token and check if it was accepted
        let accepted = matcher.accept_token(tok_idx as TokenId, None);
        tracing::info!("Step {}: token {} accepted={}", step, tok_idx, accepted);

        if !accepted {
            // Token was not accepted, something went wrong
            tracing::warn!("Token {} was not accepted, breaking", tok_idx);
            break;
        }
    }

    // Decode the sampled tokens back to text using tokenizers crate
    // Convert i32 to u32 for tokenizer.decode
    let decoded =
        tokenizer.decode(&sampled, false).map_err(|e| format!("Tokenizer decode error: {}", e))?;

    Ok(decoded)
}

#[test]
fn test_simple_generation() {
    use xgrammar::GrammarCompiler;

    let regex_pattern = r"apple|orange";

    let tokenizer_info =
        TokenizerInfo::from_pretrained(GPT_OSS_20B_PRETRAINED_ID, None, None, None)
            .expect("Failed to load tokenizer info");

    let compiler = GrammarCompiler::new(&tokenizer_info);
    let compiled_grammar = compiler.compile_regex(regex_pattern).expect("Failed to compile regex");

    let mut matcher = GrammarMatcher::with(&compiled_grammar, None, Some(true), None);

    let generated_text = generate_dummy_text(&mut matcher, GPT_OSS_20B_PRETRAINED_ID, 20, 123)
        .expect("Failed to generate text");

    let trimmed = generated_text.trim();
    assert!(
        trimmed == "apple" || trimmed == "orange",
        "Generated text should be either 'apple' or 'orange', but got: '{}'",
        trimmed
    );
}

# CS336 Spring 2025 Assignment 1: Basics Section 6: Decoding
# Deliverable: Implement a function to decode from your language model. We recommend that you
# support the following features:
# • Generate completions for a user-provided prompt (i.e., take in some x1...t and sample a completion
# until you hit an <|endoftext|> token).
# • Allow the user to control the maximum number of generated tokens.
# • Given a desired temperature value, apply softmax temperature scaling to the predicted next-word distributions before sampling.
# • Top-p sampling (Holtzman et al., 2020; also referred to as nucleus sampling), given a user-specified threshold value.

from config import Config
from transformer_lm import TransformerLM
import torch
from training_utils import load_checkpoint, AdamW
from tokenizer import Tokenizer

def main():
    config = Config()
    
    max_tokens = 100
    temperature = 0.5
    top_p = 0.9


    # load in tokenizer from pretrained
    eot = "<|endoftext|>"
    tokenizer = Tokenizer.from_files("trained_artifacts/tinystories_vocab.pkl", "trained_artifacts/tinystories_merges.pkl", special_tokens=[eot])
    eot_b = "<|endoftext|>".encode("utf-8")
    eot_id = next(i for i, b in tokenizer.vocab.items() if b == eot_b)

    # create a dummy model and load in model weights
    model = TransformerLM(
        config.vocab_size,
        config.context_length,
        config.d_model,
        config.num_layers,
        config.num_heads,
        config.d_ff,
        config.theta,
        device=torch.device(config.device),
        dtype=config.torch_dtype,
    )
    optimizer = AdamW(model.parameters())
    # it = load_checkpoint(config.model_path, model, optimizer=optimizer)

    # forward pass
    model.eval()
    prompt = "Hello, how are you?"
    x = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=torch.device(config.device)).unsqueeze(0)
    
    out_ids = generate(model, x, max_tokens, temperature, top_p, eot_id)[0]

    out_text = tokenizer.decode(out_ids.tolist())
    print(out_text)


def top_p_sampling(logits: torch.Tensor, top_p: float) -> int:
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
    keep = cumsum_probs <= top_p
    keep[0] = True  # nucleus: always keep the largest mass; avoid empty mask
    probs_trun = sorted_probs * keep.to(sorted_probs.dtype)
    probs_trun_final = probs_trun / probs_trun.sum().clamp(min=1e-10)
    sample_id = torch.multinomial(probs_trun_final, num_samples=1)
    ori_id = sorted_indices[sample_id]
    return ori_id.item()


def generate(model: torch.nn.Module, x: torch.Tensor, max_tokens: int, temperature: float, top_p: float, eot_id: int) -> str:
    with torch.no_grad():
        for _ in range(max_tokens):
            x_in = x[:, -model.context_length :] if x.size(1) > model.context_length else x
            logits = model(x_in) #(1, S, V)
            next_logits = logits[0, -1, :] # (V,)
            if temperature == 0.0:
                next_id = torch.argmax(next_logits).item()
            else:
                scaled_logits = next_logits / float(temperature)
                next_id = top_p_sampling(scaled_logits, top_p)
            if next_id == eot_id:
                break
            # concat next_id to x

            x = torch.concat([x, torch.tensor(next_id, dtype=torch.long, device=x.device).unsqueeze(0).unsqueeze(0)], dim=1)
            # print(x.shape)
    return x
    








if __name__ == "__main__":
    main()
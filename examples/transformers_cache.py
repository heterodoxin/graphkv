"""Minimal Transformers cache example.

Run from the repository root after installing `.[transformers]`.
The model must already be available locally if you use `local_files_only=True`.
"""

import torch
from transformers import AutoModelForCausalLM

from graphkv import quantize_hf_cache


MODEL_ID = "hf-internal-testing/tiny-random-GPT2LMHeadModel"


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, local_files_only=True).to(device).eval()
    input_ids = torch.arange(64, device=device).unsqueeze(0) % model.config.vocab_size
    next_id = torch.tensor([[3]], device=device)

    with torch.no_grad():
        prefill = model(input_ids, use_cache=True)

    cache = quantize_hf_cache(
        prefill.past_key_values,
        profile="graphkv-int2-max",
        output="dynamic",
        model_config=model.config,
    )

    with torch.no_grad():
        out = model(next_id, past_key_values=cache, use_cache=True)
    print(out.logits.shape)


if __name__ == "__main__":
    main()

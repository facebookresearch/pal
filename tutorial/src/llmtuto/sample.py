"""
License
-------
This source code is licensed under the CC license found in the LICENSE file
in the root directory of this source tree.

@ 2024, Meta
"""

import torch
import torch.nn.functional as F

# --------------------------------------------------------------------------------
# Language Generation
# --------------------------------------------------------------------------------


# encode sentence to tokens indices
def language_generator(prompts, tokenizer, model, seq_len, random=False, temperature=1.0, device="cuda"):
    """
    Generate new tokens from the model given a prompt.

    Parameters
    ----------
    prompts: str or list of str
        prompt(s) to complete with the model
    tokenizer: class with encode and decode methods
        tokenizer to encode the prompt and decode the model output
    model: CausalTransformer
        causal transformer model that generates new tokens
    seq_len: int
        number of tokens to generate
    random: bool, default is False
        whether to sample from the distribution or take the most likely token
    temperature: float, default is 1.
        temperature of the softmax distribution. Only used if random is True
    device: str, default is 'cuda'
        device to use for the model

    Returns
    -------
    list of str
        list of generated setences
    """
    if isinstance(prompts, str):
        prompts = [prompts]

    if random:

        def choose_token_from_logit(logits):
            logits /= temperature
            probs = F.softmax(logits, dim=-1)
            return torch.multinomial(probs, num_samples=1)

    else:

        def choose_token_from_logit(logits):
            return torch.argmax(logits, dim=-1, keepdim=True)

    tokens = [tokenizer.encode(seq) for seq in prompts]

    # handling different sentences lengths
    lenghts = [len(seq) for seq in tokens]
    nb_sentences, max_len, min_len = len(prompts), max(lenghts), min(lenghts)

    seq_idx = torch.zeros((nb_sentences, max_len), dtype=torch.long, device=device)
    mask = torch.zeros((nb_sentences, max_len - min_len), dtype=torch.bool, device=device)
    for i, seq in enumerate(tokens):
        seq_idx[i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
        mask[i, : len(seq) - min_len] = 1

    for i in range(min_len, max_len):
        logits = model(seq_idx[:, :i])[:, -1, :]
        next_token = choose_token_from_logit(logits).squeeze()
        torch.where(mask[:, i - min_len], seq_idx[:, i], next_token, out=seq_idx[:, i])

    # (inefficient) generation of new tokens
    with torch.no_grad():
        for i in range(seq_len):
            logits = model(seq_idx)[:, -1, :]
            next_token = choose_token_from_logit(logits)
            seq_idx = torch.cat([seq_idx, next_token], dim=-1)

    return [tokenizer.decode(list(seq)) for seq in seq_idx]

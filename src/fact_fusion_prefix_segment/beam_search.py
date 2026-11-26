"""Batched beam search with compact, variable-length fact prefixes."""

import torch
from torch.nn import functional as F


def pack_prefix_and_caption(prefix_ids, prefix_lengths, sequences, pad_idx):
    """Replace the caption's initial SOS with the supplied prefix, without interior pads."""
    lengths = prefix_lengths + sequences.size(1) - 1
    positions = torch.arange(int(lengths.max()), device=sequences.device).unsqueeze(0)
    positions = positions.expand(sequences.size(0), -1)
    prefix_positions = positions.clamp(max=prefix_ids.size(1) - 1)
    caption_positions = (positions - prefix_lengths.unsqueeze(1) + 1).clamp(0, sequences.size(1) - 1)
    input_ids = torch.where(
        positions < prefix_lengths.unsqueeze(1),
        prefix_ids.gather(1, prefix_positions), sequences.gather(1, caption_positions),
    )
    attention_mask = positions < lengths.unsqueeze(1)
    return input_ids.masked_fill(~attention_mask, pad_idx), attention_mask, lengths


@torch.no_grad()
def beam_search(decoder, memory, vocab, prefix_ids, prefix_lengths, beam_size=5, max_length=40,
                length_penalty=1.0, memory_key_padding_mask=None):
    if not 1 <= beam_size <= len(vocab) or max_length < 2:
        raise ValueError("Invalid beam size or caption max_length.")
    sep_idx = vocab.sep_idx()
    if sep_idx is None:
        raise ValueError("Prefix vocabulary must contain the SEP special token.")
    batch_size = memory.size(0)
    device = memory.device
    if prefix_ids.size(0) != batch_size or prefix_lengths.shape != (batch_size,):
        raise ValueError("Prefix batch must align with memory.")
    if ((prefix_lengths < 1) | (prefix_lengths > prefix_ids.size(1))).any():
        raise ValueError("Invalid prefix length.")
    if not prefix_ids.gather(1, (prefix_lengths - 1).unsqueeze(1)).eq(vocab.sos_idx()).all():
        raise ValueError("Every prefix must end with SOS.")
    memory = memory.repeat_interleave(beam_size, dim=0)
    prefix_ids = prefix_ids.repeat_interleave(beam_size, dim=0)
    prefix_lengths = prefix_lengths.repeat_interleave(beam_size, dim=0)
    if memory_key_padding_mask is not None:
        memory_key_padding_mask = memory_key_padding_mask.repeat_interleave(beam_size, dim=0)

    sequences = torch.full((batch_size * beam_size, 1), vocab.sos_idx(), device=device, dtype=torch.long)
    scores = memory.new_full((batch_size * beam_size,), float("-inf"), dtype=torch.float32)
    scores[torch.arange(batch_size, device=device) * beam_size] = 0
    finished = torch.zeros_like(scores, dtype=torch.bool)
    caption_lengths = torch.ones_like(scores)
    for _ in range(max_length - 1):
        input_ids, attention_mask, lengths = pack_prefix_and_caption(
            prefix_ids, prefix_lengths, sequences, vocab.pad_idx(),
        )
        logits = decoder(
            input_ids=input_ids, memory=memory, prefix_lengths=prefix_lengths,
            attention_mask=attention_mask, memory_key_padding_mask=memory_key_padding_mask,
        )
        last_logits = logits[torch.arange(logits.size(0), device=device), lengths - 1]
        log_probs = F.log_softmax(last_logits.float(), dim=-1)
        log_probs[:, sep_idx] = float("-inf")
        log_probs[:, vocab.sos_idx()] = float("-inf")
        log_probs[:, vocab.pad_idx()] = float("-inf")
        log_probs[finished] = float("-inf")
        log_probs[finished, vocab.pad_idx()] = 0
        candidates = scores.unsqueeze(1) + log_probs
        scores, flat_indices = candidates.view(batch_size, -1).topk(beam_size, dim=1)
        parent_indices = flat_indices // len(vocab)
        token_ids = flat_indices % len(vocab)
        parents = (torch.arange(batch_size, device=device).unsqueeze(1) * beam_size + parent_indices).flatten()
        sequences = torch.cat([sequences[parents], token_ids.reshape(-1, 1)], dim=1)
        scores = scores.flatten()
        finished = finished[parents] | token_ids.flatten().eq(vocab.eos_idx())
        caption_lengths = caption_lengths[parents] + (~finished).to(caption_lengths.dtype)
        # Prefixes and memory are identical across all beams of the same image.
        if finished.all():
            break
    if length_penalty > 0:
        scores = scores / caption_lengths.pow(length_penalty)
    best = scores.view(batch_size, beam_size).argmax(dim=1)
    best = torch.arange(batch_size, device=device) * beam_size + best
    output = sequences[best]
    return F.pad(output, (0, max_length - output.size(1)), value=vocab.pad_idx())

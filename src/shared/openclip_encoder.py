"""Load matching OpenCLIP weights, image preprocessing and text tokenization."""
import open_clip

def load_openclip_retrieval_components(model_id: str, *, device="cpu", precision="fp32"):
    """Load a checkpoint using its full OpenCLIP ID, e.g. ``hf-hub:org/repo``."""

    model, preprocess = open_clip.create_model_from_pretrained(
        model_id, device=device, precision=precision,
    )
    model.eval().requires_grad_(False)
    tokenizer = open_clip.get_tokenizer(model_id, context_length=model.context_length)
    return model, preprocess, tokenizer

from transformers import AutoTokenizer, AutoModel
import torch
import config

_tokenizer = None
_model = None

def _load_model():
    """Loads once, reused across every call — loading a model from disk is
    slow, you don't want to repeat it per filing."""
    global _tokenizer, _model
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(config.EMBEDDING_MODEL)
        _model = AutoModel.from_pretrained(config.EMBEDDING_MODEL)
        _model.eval()  # inference mode, not training


def chunk_tokens(text: str) -> list[list[int]]:
    """Tokenize text, split into 500-token windows (leaving room for
    [CLS]/[SEP])."""
    token_ids = _tokenizer(text, add_special_tokens=False)["input_ids"]

    chunk_size = config.MAX_TOKENS - 2
    chunks = [
        token_ids[i:i + chunk_size]
        for i in range(0, len(token_ids), chunk_size)  # step by chunk_size
    ]
    return chunks


def embed_chunk(token_ids: list[int]) -> torch.Tensor:
    """Add special tokens, run through the model, mean-pool into one 384-dim vector."""
    # Manually wrap the chunk with [CLS] ... [SEP]
    input_ids = [_tokenizer.cls_token_id] + token_ids + [_tokenizer.sep_token_id]
    attention_mask = [1] * len(input_ids)  # every token here is real, no padding within one chunk

    inputs = {
        "input_ids": torch.tensor([input_ids]),       # [1, seq_len], the leading 1 is the "batch size"
        "attention_mask": torch.tensor([attention_mask]),  # same shape, all 1s
    }

    with torch.no_grad():
        output = _model(**inputs)

    token_embeddings = output.last_hidden_state
    mask = inputs["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = torch.sum(token_embeddings * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return (summed / counts).squeeze(0)


def embed_text(text: str) -> list[float] | None:
    """
    Full pipeline for one section's text:
    1. If len(text) < config.MIN_CHARS_FOR_EMBEDDING (or text is None), return None
    2. _load_model() to ensure it's ready
    3. chunk_tokens(text) -> list of chunks
    4. embed_chunk() each chunk, collect the vectors
    5. Average all chunk vectors together (torch.stack(...).mean(dim=0))
    6. Convert to a plain Python list (.tolist()) so it's JSON-serializable
    """
    if len(text) < config.MIN_CHARS_FOR_EMBEDDING or text is None:
        return None
    _load_model()
    list_chunks = chunk_tokens(text)
    vectors = [embed_chunk(chunk) for chunk in list_chunks]

     # return average chunk vectors, convert to Python list so it is JSON-serializable
    return torch.stack(vectors).mean(dim=0).tolist()


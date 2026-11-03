"""Convert guest descriptions into vectors with a language model that runs locally.

Descriptions never leave this computer. The model is downloaded once, on first use.
"""
from pathlib import Path
from threading import Lock

import numpy as np

MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
MODEL_CACHE = Path(__file__).parent / '.model_cache'

_model = None
_lock = Lock()


class EmbeddingUnavailable(Exception):
    """The language model couldn't be loaded or run."""


def embed_texts(texts):
    """Return an array with one row for each text."""
    global _model
    with _lock:
        try:
            if _model is None:
                # Imported here because it's slow to load and tests don't need it.
                from fastembed import TextEmbedding
                _model = TextEmbedding(model_name=MODEL_NAME, cache_dir=str(MODEL_CACHE))
            return np.array(list(_model.embed(texts)))
        except Exception as error:
            raise EmbeddingUnavailable(str(error)) from error

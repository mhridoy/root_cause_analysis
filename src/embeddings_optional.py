"""
OPTIONAL deep-learning upgrade: pretrained language-model sentence embeddings.

On ~360 reports, a from-scratch deep net does not beat a linear model (we
measured it). The way deep learning actually helps at this data scale is via
TRANSFER LEARNING -- embeddings from a model pretrained on billions of words.
This module provides those embeddings IF the libraries are installed, and fails
gracefully otherwise (the core system never depends on it).

Enable on your own machine (needs internet for the first download):

    pip install sentence-transformers

Then build embedding features and feed them to the same MLP/linear trainer:

    from src.embeddings_optional import embed_texts, is_available
    if is_available():
        X = embed_texts(list_of_report_texts)   # dense (n, 384) features
        # ... train MLPClassifier / SoftmaxClassifier on X vs grounded labels ...

A clinical-domain model (e.g. 'emilyalsentzer/Bio_ClinicalBERT' via the
transformers library) will usually beat a general one for medical text.
"""
from functools import lru_cache

_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def is_available():
    try:
        import sentence_transformers  # noqa: F401
        return True
    except Exception:
        return False


@lru_cache(maxsize=2)
def _load(model_name):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def embed_texts(texts, model_name=_DEFAULT_MODEL, batch_size=16):
    """Return dense embeddings (numpy array) for a list of texts.
    Raises RuntimeError with guidance if the optional dependency is missing."""
    if not is_available():
        raise RuntimeError(
            "sentence-transformers is not installed. This is an OPTIONAL deep "
            "upgrade. Install with: pip install sentence-transformers")
    model = _load(model_name)
    return model.encode(list(texts), batch_size=batch_size,
                        show_progress_bar=False, normalize_embeddings=True)

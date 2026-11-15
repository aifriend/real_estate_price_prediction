"""Shared test setup.

`src/service/EmbeddingService.py` imports `sentence_transformers`, which is not
declared in pyproject.toml. The tests never load an embedding model, so when the
package is missing a stub module is registered to let `TrainService` import.
"""
import sys
import types

try:
    import sentence_transformers  # noqa: F401
except ImportError:
    stub = types.ModuleType('sentence_transformers')
    stub.SentenceTransformer = object
    sys.modules['sentence_transformers'] = stub

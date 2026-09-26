"""Chat with the posteriors of the example notebooks using a local Gemma model.

    ollama pull gemma4:e2b-it-qat
    uv run --extra reports panel serve reports/explorer_app.py --show

POSTERIOR_LLM picks another Ollama model, e.g. POSTERIOR_LLM=gemma4:e4b-it-qat.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from posterior_lumen.app import make_explorer  # noqa: E402

make_explorer().servable()

"""ExpHandler — Experiment management GUI.

Run locally:
    python app.py

Access in browser at http://127.0.0.1:8050
"""

import os
import sys

# Ensure project root is on the path (important when running from other dirs)
_root = os.path.dirname(os.path.abspath(__file__))
if _root not in sys.path:
    sys.path.insert(0, _root)

from dash import Dash
from layout import gen_layout
import callbacks  # noqa — registers all callbacks by importing the package

app = Dash(
    __name__,
    title="ExpHandler",
    suppress_callback_exceptions=True,  # needed for dynamically rendered components
)
app.layout = gen_layout()


if __name__ == "__main__":
    host = os.getenv("HOSTNAME", "127.0.0.1")
    port = int(os.getenv("PORT", 8050))
    app.run(host=host, port=port, debug=True)

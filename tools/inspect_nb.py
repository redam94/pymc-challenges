"""Print the text outputs of an executed notebook and dump its figures as PNGs (for reviewing a build).

    uv run python tools/inspect_nb.py notebooks/solutions/C01_golf_putting_solution.ipynb [figure_dir]
"""

import base64
import sys
from pathlib import Path

import nbformat

nb_path = Path(sys.argv[1])
fig_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else None
nb = nbformat.read(nb_path, as_version=4)
n_fig = 0
for i, cell in enumerate(nb.cells):
    if cell.cell_type != "code":
        continue
    for out in cell.get("outputs", []):
        if out.output_type == "stream":
            text = out.text
        elif out.output_type == "error":
            text = f"ERROR {out.ename}: {out.evalue}"
        else:
            data = out.get("data", {})
            if "image/png" in data:
                n_fig += 1
                text = f"<figure {n_fig}>"
                if fig_dir is not None:
                    fig_dir.mkdir(parents=True, exist_ok=True)
                    (fig_dir / f"{nb_path.stem}_{n_fig:02d}.png").write_bytes(base64.b64decode(data["image/png"]))
            else:
                text = data.get("text/markdown") or data.get("text/plain", "")
        text = text.strip()
        if text:
            print(f"--- cell {i} [{out.output_type}{'/' + out.name if out.output_type == 'stream' else ''}]")
            print(text[:1500])

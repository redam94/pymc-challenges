"""An interactive report over the posteriors of five example notebooks.

    uv run --extra reports panel serve reports/report_app.py --show

Summaries come from the local Ollama model (POSTERIOR_LLM, default gemma4:e2b-it-qat);
POSTERIOR_LLM=none builds the same report with no LLM at all.

The report is a Lumen ``Report`` (built in ``posterior_lumen/report.py``) and is executed
by Lumen, but it is displayed in a page of our own. Served on its own, Lumen 1.3's Report
widget never signals "document ready" to Panel (even an empty Report), so its play button
and every deferred callback never run and the page stays at "ready to launch".
"""

import io
import os
import sys
from pathlib import Path

import panel as pn
import panel_material_ui as pmui

sys.path.insert(0, str(Path(__file__).resolve().parent))
from posterior_lumen.report import make_report  # noqa: E402

pn.extension("plotly", "tabulator")

CAPTIONS = os.environ.get("POSTERIOR_LLM", "").lower() != "none"

report = make_report(captions=CAPTIONS, auto_execute=False)
sections = list(report)
# One tab per model. (Filling an expanded Material UI Card with Bokeh plots after load broke
# Bokeh's layout pass - "Cannot read properties of undefined (reading 'top_panel')" - so the
# sections live in plain Panel columns inside tabs.)
cards = [pn.Column(pmui.Typography("Waiting to run…", variant="body2"), sizing_mode="stretch_width")
         for _ in sections]
tabs = pn.Tabs(*[(s.title, c) for s, c in zip(sections, cards)], dynamic=True, sizing_mode="stretch_width")
status = pmui.Typography("", variant="body2", sx={"color": "text.secondary"})
progress = pmui.LinearProgress(value=0, variant="determinate", sizing_mode="stretch_width", visible=False)


def _outputs(task):
    """A task's rendered outputs; its title becomes a subheading."""
    items = [pmui.Typography(task.title, variant="h5", margin=(18, 0, 6, 0))] if task.title else []
    if not task.title and items == [] and type(task).__name__ == "Story":
        items.append(pmui.Divider(margin=(20, 0)))
    for view in task.views:
        if view is getattr(task, "_title", None):
            continue
        if type(view).__name__ == "SQLEditor":  # Lumen's editor squashes to two rows here
            data = view.component.data
            items += [pn.widgets.Tabulator(data, show_index=False, disabled=True, sizing_mode="stretch_width",
                                           height=min(420, 36 * len(data) + 44), layout="fit_data_table"),
                      pmui.Accordion(("SQL behind this table", pn.pane.Markdown(f"```sql\n{view.spec}\n```")),
                                     sizing_mode="stretch_width")]
            continue
        rendered = task._render_output(view)
        if rendered is not None:
            items.append(rendered)
    if task.status == "error":
        items.append(pmui.Alert(object="This part failed; see the server log.", alert_type="error"))
    return items


async def run(event=None):
    run_button.disabled = True
    progress.visible, progress.value = True, 0
    report.reset()
    total = sum(len(s) for s in sections)
    done = 0
    context = {}
    for card, section in zip(cards, sections):
        card[:] = [pmui.Typography("Running…", variant="body2")]
        body = []
        for task in section:
            status.object = f"{section.title}: {task.title}"
            try:
                _, out = await task.execute(context)
                context.update(out or {})
            except Exception as e:  # keep going: one failing view should not blank the report
                import traceback
                traceback.print_exc()
                task.status = "error"
                body.append(pmui.Alert(object=f"{task.title}: {e}", alert_type="error"))
            body += _outputs(task)
            done += 1
            progress.value = int(100 * done / total)
        card[:] = body
    status.object = "Done. Move the widgets to explore; the summaries under 'In short' were written " + (
        "by the local LLM from the computed facts." if CAPTIONS else "without an LLM (facts only).")
    progress.visible = False
    run_button.disabled = False


def notebook():
    return io.StringIO(report.to_notebook())


run_button = pmui.Button(label="Run again", icon="refresh", on_click=run, variant="outlined")
download = pmui.FileDownload(callback=notebook, filename="posterior_report.ipynb", label="Download as notebook",
                             variant="outlined")

page = pmui.Page(
    title="Posterior Report",
    sidebar=[
        pmui.Typography("Five Bayesian models, explored from their posterior draws.", variant="body1"),
        pmui.Typography("Every number is computed from the draws; the LLM only phrases the summaries.",
                        variant="body2", sx={"color": "text.secondary"}),
        run_button, download,
    ],
    main=[pn.Column(status, progress, tabs, sizing_mode="stretch_width", max_width=1300)],
)
pn.state.onload(run)
page.servable()

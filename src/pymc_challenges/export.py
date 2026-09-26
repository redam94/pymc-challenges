"""Save a notebook's full posterior for the Lumen reports in ``reports/``.

The JSON exports in ``.scratch/artifact`` carry a few hundred draws for a web page. This
writes everything instead: the fitted posterior (every chain and draw of every parameter)
and the notebook's derived quantities over all the draws it computed them on, as one
netCDF file with two groups, ``posterior`` and ``derived``::

    from pymc_challenges.export import save_posterior
    save_posterior("E40", idata, {
        "clinton_state_share": (("sample", "state"), final, {"state": STATE_NAMES_LIST},
                                "share 0-1", "Clinton two-party share on election day"),
        "clinton_electoral_votes": (("sample",), ev_c, {}, "electoral votes", "..."),
    })

Every derived array's first dimension is ``sample`` (equally likely draws); quantities may
have different numbers of draws, so each gets its own group ``derived/<name>``. A dimension
listed in ``x_dims`` is continuous (spend, age, ...) and becomes the report's x axis.
Parameters with more than ``max_elements`` values per draw (observation-level
deterministics) are left out: they are data-sized, not parameters anyone browses.
"""

from pathlib import Path

import numpy as np
import xarray as xr


def repo_root():
    here = Path.cwd()
    return next(p for p in [here, *here.parents] if (p / "pyproject.toml").exists())


def save_posterior(model_id, idata, derived, var_names=None, x_dims=(), max_elements=5000):
    """Write ``reports/posteriors/<model_id>.nc`` and return its path.

    ``derived`` maps a name to ``(dims, values, coords, unit, meaning)``. Parameters are
    stored as float32 (plenty for summaries and plots, half the size).
    """
    post = idata.posterior
    post = post.to_dataset() if hasattr(post, "to_dataset") else post
    if var_names is not None:
        post = post[var_names]
    per_draw = {v: int(np.prod([n for d, n in post[v].sizes.items() if d not in ("chain", "draw")]))
                for v in post.data_vars}
    dropped = [v for v, n in per_draw.items() if n > max_elements]
    post = post.drop_vars(dropped).astype("float32")
    post.attrs = {k: v for k, v in post.attrs.items() if isinstance(v, (str, int, float))}

    groups = {"posterior": post}
    for name, (dims, values, coords, unit, meaning) in derived.items():
        values = np.asarray(values, dtype="float64")
        assert dims[0] == "sample" and values.shape and values.ndim == len(dims), \
            f"{name}: dims {dims} vs shape {values.shape}"
        da = xr.DataArray(values, dims=dims, coords={d: np.asarray(coords[d]) for d in dims[1:] if d in coords},
                          attrs={"unit": unit, "meaning": meaning})
        groups[f"derived/{name}"] = xr.Dataset({name: da})
    groups["derived"] = xr.Dataset(attrs={"x_dims": ",".join(x_dims), "model_id": model_id})

    out = repo_root() / "reports" / "posteriors" / f"{model_id}.nc"
    out.parent.mkdir(parents=True, exist_ok=True)
    xr.DataTree.from_dict(groups).to_netcdf(out, engine="h5netcdf")
    n_draws = post.sizes.get("chain", 1) * post.sizes.get("draw", 1)
    print(f"wrote {out.relative_to(repo_root())}: {len(post.data_vars)} parameters x {n_draws} draws"
          + (f" (left out, too large: {', '.join(dropped)})" if dropped else "")
          + f"; {len(derived)} derived quantities; {out.stat().st_size / 1e6:.1f} MB")
    return out

import numpy as np
import tempfile
import shutil
import semopy
import os

from .format import format_p

# it uses graphviz. for windows, use https://graphviz.gitlab.io/download/
# https://arxiv.org/pdf/2106.01140.pdf
# formula = f"""
# ПП =~ {' + '.join(p)}
# НЭ =~ {' + '.join(e)}
# ПП ~ НЭ + {' + '.join(x)} + Группа
# """


def sem(
    data,
    formula,
    method="MLW",  # MLW ULS GLS FIML DWLS WLS
    solver="SLSQP",
    bootstrap=100,
    plot_covs=True,
    se_robust=False,
    standardized=True,
    format_pval=True,
    save_to_file=None,
    return_fig=False,
    decimal=3,
    seed=13,
):
    np.random.seed(seed)
    model = semopy.Model(formula)
    model.fit(data, obj=method, solver=solver)
    if bootstrap:
        semopy.bias_correction(model, n=bootstrap, resample_mean=True)
    metrics = (
        semopy.calc_stats(model).T.round(decimal).rename(columns={"Value": "value"})
    )
    stats = model.inspect(std_est=standardized, se_robust=se_robust)
    stats.columns = [i.lower().replace(". ", "_") for i in stats.columns]
    if format_pval:
        stats["p-value"] = stats["p-value"].apply(
            lambda x: format_p(x, use_letter=False, keep_spaces=False, no_equals=True)
        )
    stats = stats.replace("-", np.nan).infer_objects(copy=False).round(decimal)
    if return_fig or save_to_file:
        fig = _semplot(
            model,
            save_to_file,
            plot_covs,
            standardized,
        )
        if return_fig:
            return stats, metrics, fig
    return stats, metrics


def _semplot(model, save_to_file, plot_covs, std_ests):
    tmp_file = os.path.join(tempfile.gettempdir(), "sem_tmp_fig.png")
    fig = semopy.semplot(
        model, tmp_file, plot_covs=plot_covs, std_ests=std_ests, show=False
    )
    if isinstance(save_to_file, str):
        _, ext = os.path.splitext(save_to_file)
        fig.render(tmp_file, view=False, format=ext[1:], quiet=True)
        shutil.copy(tmp_file + ext, save_to_file)
    return fig

import numpy as np
import pandas as pd
import semopy
import os


from .format import format_p

# it uses graphviz. for windows, use https://graphviz.gitlab.io/download/
# https://arxiv.org/pdf/2106.01140.pdf
# https://semopy.com/cite.html
# https://gitlab.com/georgy.m/semopy
# ~ to specify structural part,
# =~ to specify measurement part,
# ~~ to specify common variance between variables.
# formula = f"""
# x1 ~ x2 + x3
# x3 ~ x2 + eta1
# eta1 =~ y1 + y2 + y3
# eta1 ~ x1
# """


def sem(
    data,
    formula,
    method="MLW",  # MLW ULS GLS FIML DWLS WLS
    solver="SLSQP",
    bootstrap=100,
    se_robust=True,  # Huber-White correction
    return_model=False,
    seed=13,
):
    np.random.seed(seed)
    model = semopy.Model(formula)
    model.fit(data, obj=method, solver=solver)
    if bootstrap:
        semopy.bias_correction(model, n=bootstrap, resample_mean=True)
    metrics = semopy.calc_stats(model).T.rename(columns={"Value": "value"})
    metrics = pd.concat([pd.Series({"N": data.shape[0]}, name="value"), metrics])
    stats = model.inspect(std_est=True, se_robust=se_robust)
    stats.columns = [i.lower().replace(". ", "_") for i in stats.columns]
    stats = stats.replace("-", np.nan).infer_objects(copy=False)
    if return_model:
        return stats, metrics, model
    return stats, metrics


def sem_report(stats, metrics, decimal=3, format_pval=True):
    # https://people.ucsc.edu/~zurbrigg/psy214b/09SEM8a.pdf
    # χ2(48, N = 500) = 303.80, p < .001, TLI = .86, CFI = .90

    stats = stats.copy()
    if format_pval:
        stats["p-value"] = (
            stats["p-value"]
            .fillna("")
            .apply(
                lambda x: format_p(
                    x, use_letter=False, keep_spaces=False, no_equals=True
                )
            )
        )
    r = metrics.iloc[:6].T.to_dict()
    m = metrics.iloc[6:11].T.to_dict()
    k = metrics.iloc[11:].T.to_dict()
    r = f"χ2({int(r['DoF']['value'])}, N = {int(r['N']['value'])}) = {round(r['chi2']['value'], decimal)}, {format_p(r['chi2 p-value']['value'])}"
    m = ", ".join([f"{i} = " + f"{round(v['value'], decimal)}"[1:] for i, v in m.items()])
    k = ", ".join([f"{i} = " + f"{round(v['value'], decimal)}" for i, v in k.items()])
    stats = stats.round(decimal)
    stats.loc[stats.index[0], "model"] = r + ", " + m + ", " + k
    return stats


def sem_plot(
    stats,
    save_to_file,
    plot_covs=True,
    std_ests=True,
    format_pval=True,
    format_fig="png",
    method="semopy",
    return_fig=True,
    show_fig=False,
    decimal=3,
    dpi=600,
    seed=13,
):
    #

    np.random.seed(seed)
    elements = stats[stats["op"] == "~"]
    latent = set(elements["rval"].to_list())
    observed = set(elements["lval"].to_list()) - latent

    if method == "semopy":
        import tempfile
        import shutil

        tfile = os.path.join(tempfile.gettempdir(), f"sem_tmp_fig.{format_fig}")
        inspection = stats.copy()
        inspection["Estimate"] = inspection["estimate"]
        inspection["Est. Std"] = inspection["est_std"]
        inspection["p-value"] = inspection["p-value"].fillna("-")
        model = semopy.Model("")
        model.last_result = None
        model.vars["latent"] = latent
        model.vars["observed"] = observed
        model.vars["all"] = list(latent) + list(observed)
        fig = semopy.semplot(
            model,
            tfile,
            plot_covs=plot_covs,
            std_ests=std_ests,
            inspection=inspection,
            show=False,
        )
        shutil.copy(tfile, save_to_file)
        if return_fig:
            return fig

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
    r = f"χ2({int(r['DoF']['value'])}, N = {int(r['N']['value'])}) = {round(r['chi2']['value'], decimal)}, {format_p(r['chi2 p-value']['value'])}"
    m = ", ".join(
        [f"{i} = " + f"{round(v['value'], decimal)}"[1:] for i, v in m.items()]
    )
    r = r + ", " + m
    m = metrics.iloc[11:].T.to_dict()
    m = ", ".join([f"{i} = " + f"{round(v['value'], decimal)}" for i, v in m.items()])
    stats = stats.round(decimal)
    stats.loc[stats.index[0], "model"] = r + ", " + m
    return stats


def sem_plot(
    stats,
    save_to_file,
    plot_covs=True,
    std_ests=True,
    format_pval=True,
    format_fig="pdf",
    method="semopy",
    return_fig=True,
    show_fig=False,
    decimal=3,
    dpi=600,
    model=None,
):
    #
    if method == "semopy":
        if model is None:
            raise AssertionError(
                "For the semopy method, the model parameter is necessary!"
            )
        import tempfile
        import shutil

        tfile = os.path.join(tempfile.gettempdir(), f"sem_tmp_fig.{format_fig}")
        inspection = stats.copy()
        inspection["Estimate"] = inspection["estimate"]
        inspection["Est. Std"] = inspection["est_std"]
        inspection["p-value"] = inspection["p-value"].fillna("-")
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


'''
def _semplot(model, save_to_file, plot_covs, std_ests):
import networkx as nx
    tmp_file = os.path.join(tempfile.gettempdir(), "sem_tmp_fig.png")
    fname, ext = os.path.splitext(tmp_file)
    fig = semopy.semplot(
        model, tmp_file, plot_covs=plot_covs, std_ests=std_ests, show=False
    )
    from graphviz import Source

    g = nx.nx_pydot.read_dot(fname)
    for e in g.edges:
        print(e)
        for k, v in g.edges[e].items():
            if k == "label" and "p-val" in v:
                g.edges[e]["label"] = g.edges[e]["label"].replace("p-val", "p")
    # import matplotlib.pyplot as plt
    # nx.draw(g, pos=nx.spring_layout(g))  # use spring layout
    # plt.show()
    """
    if return_fig or save_to_file:
        fig = _semplot(
            model,
            save_to_file,
            plot_covs,
            standardized,
        )
    if return_fig:
            return stats, metrics, fig
    g_ = nx.nx_pydot.to_pydot(g)
    #print(g_)
    fig = Source(str(g_))
    
    """
    if isinstance(save_to_file, str):
        _, ext = os.path.splitext(save_to_file)
        fig.render(tmp_file, view=False, format=ext[1:], quiet=True)
        shutil.copy(tmp_file + ext, save_to_file)
    return fig
'''

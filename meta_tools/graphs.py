import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
import numpy as np
import pandas as pd

def radar_factory(num_vars, frame='circle'):
    """Create a radar chart with `num_vars` axes.
    This function creates a RadarAxes projection and registers it.
    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle' | 'polygon'}
        Shape of frame surrounding axes.
    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
    
    class RadarTransform(PolarAxes.PolarTransform):
        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):
        name = 'radar'
        PolarTransform = RadarTransform
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

        def draw(self, renderer):
            """ Draw. If frame is polygon, make gridlines polygon-shaped """
            if frame == 'polygon':
                gridlines = self.yaxis.get_gridlines()
                for gl in gridlines:
                    gl.get_path()._interpolation_steps = num_vars
            super().draw(renderer)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)
    register_projection(RadarAxes)
    return theta

def ols_tree_graph(r, title, robust=False):
    def addlabels(x, y, sig):
      for i, j in enumerate(x):
          if j < 0:
            j -= 0.1
          else:
            j += 0.05
          plt.text(j, y[i], sig[i])
    rb = '_robust' if robust else ''
    d = r[pd.notna(r['sig'+rb])]# & (r['sig'+rb] != '')]
    x1 = d[d['coef'+rb] < 0]['coef'+rb].to_list()
    sig1 = d[d['coef'+rb] < 0]['sig'+rb].to_list()
    x2 = d[d['coef'+rb] > 0]['coef'+rb].to_list()
    sig2 = d[d['coef'+rb] > 0]['sig'+rb].to_list()
    y1 = range(len(x1))
    y2 = range(len(x1), (len(x1) + len(x2))) 
    y = range(len(x1) + len(x2))
    yl1 = d[d['coef'+rb] < 0]['index'].to_list()
    yl2 = d[d['coef'+rb] > 0]['index'].to_list()
    fig, axes = plt.subplots(ncols=2, sharey=True, figsize=(5, 3.5))
    bar1 = axes[0].barh(y1, x1, align='center', color='red')
    bar2 = axes[1].barh(y2, x2, align='center', color='blue')
    axes[0].set(yticks=y, yticklabels=yl1 + yl2)
    axes[0].set_xlim([-1, -0.001])
    axes[1].set_xlim([0.001, 1])
    for ax in axes.flat:
        ax.margins(0.03)
        ax.grid(True)
    for i, b in enumerate(bar1):
        if len(sig1[i]):
          b.set_color('mediumorchid')
        else:
          b.set_color('thistle')
    for i, b in enumerate(bar2):
        if len(sig2[i]):
          b.set_color('mediumorchid')
        else:
          b.set_color('thistle')
    addlabels(x1, y1, sig1)
    addlabels(x2, y2, sig2)
    axes[0].tick_params(axis='y', which='both', labelleft=True, labelright=False)
    axes[1].tick_params(axis='y', which='both', labelleft=False, labelright=False)
    fig.suptitle(title, y=1.05)
    fig.text(0.5, 0, 'β', ha='center')
    #fig.tight_layout()
    fig.subplots_adjust(wspace=0, top=0.93)
    plt.show()
    return fig
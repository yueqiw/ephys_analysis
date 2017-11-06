import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
from scipy.spatial import distance
from scipy.cluster import hierarchy
import matplotlib.patches as mpatches
import io
import base64

feature_names = dict(tau='Membrane Time Constant',
                     input_resistance='Input Resistance',
                     sag='Sag',
                     cm_est='Membrane Capacitance',
                     v_rest='Resting Vm',
                    f_i_curve_slope='F-I Curve Slope',
                    ap_threshold='Spike threshold',
                    ap_width='Spike Width',
                    ap_height='Spike Height',
                    ap_peak='Spike Peak',
                    ap_trough='Spike Trough',
                    ap_trough_to_threshold='Spike Trough-to-Threshold',
                    ap_upstroke='Spike Upstroke',
                    ap_downstroke='Spike Downstroke',
                    ap_updownstroke_ratio='Upstroke-Downstroke Ratio',
                    hs_firing_rate='Firing Rate',
                    hs_latency='First Spike Latency',
                    hs_adaptation='Adaptation')

features_noAP = ['tau', 'input_resistance', 'sag', 'cm_est', 'v_rest']
features_ap = ['f_i_curve_slope', 'ap_threshold',
               'ap_width', 'ap_height', 'ap_peak', 'ap_trough',
               'ap_trough_to_threshold', 'ap_upstroke', 'ap_downstroke',
               'ap_updownstroke_ratio', 'hs_firing_rate', 'hs_latency']
features_isi = ['hs_adaptation', 'hs_median_isi']
qc_cols = ['v_baseline',
       'bias_current', 'vm_for_sag', 'rheobase_stim_amp', 'hold', 'ra_pre',
       'ra_post', 'start', 'step', 'ra_est']
id_cols = ['experiment', 'cell', 'recording', 'id', 'date']
meta_cols = ['has_ap', 'strain', 'dob', 'age', 'slicetype']

muted = {name: 'rgba(' + str(a) + ', ' + str(b) + ', ' + str(c) + ')' for name, (a, b, c) \
    in zip(['blue', 'green', 'red', 'purple', 'yellow', 'cyan'], sns.color_palette("muted"))}

def cluster_heatmap(data, feature_names, idx_color_mapping, exp_lut, legend=True):
    '''draw heatmap with hierachical clustering for all cells using Seaborn.'''
    df = pd.DataFrame(data, columns=feature_names.values()).T
    correlations_array = np.asarray(df)

    row_linkage = hierarchy.linkage(distance.pdist(correlations_array), method='average')
    col_linkage = hierarchy.linkage(distance.pdist(correlations_array.T), method='average')

    g = sns.clustermap(df, row_linkage=row_linkage, col_linkage=col_linkage, center=0,
                       col_colors = idx_color_mapping, figsize=(20,15))
    _ = plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=28)
    _ = plt.setp(g.ax_heatmap.get_xticklabels(), rotation=0, fontsize=22)

    if legend:
        legend_patches = []
        for k, v in exp_lut.items():
            legend_patches.append(mpatches.Patch(color=v, label=k))

        plt.legend(handles=legend_patches, fontsize=28)
    return g

def byte_encode_img(fig):
    '''save the figure into memory buffer and byte encode it for html.'''
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    encoded_heatmap = base64.b64encode(buf.getvalue())
    buf.close()
    # use decoded png on html
    decoded_heatmap = 'data:image/png;base64,{}'.format(encoded_heatmap.decode())
    return decoded_heatmap

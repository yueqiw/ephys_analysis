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

feature_names = dict(input_resistance='Input Resistance (MOhm)',
                     sag='Sag',
                     capacitance='Membrane Capacitance (pF)',
                     v_rest='Resting Vm (mV)',
                    f_i_curve_slope='F-I Curve Slope',
                    ap_threshold='Spike threshold (mV)',
                    ap_width='Spike Width (ms)',
                    ap_peak_to_threshold='Spike Amplitude (mV)',
                    ap_trough_to_threshold='AHP Amplitude (mV)',
                    ap_upstroke='Spike Upstroke (mV/ms)',
                    ap_updownstroke_ratio='Upstroke-Downstroke Ratio',
                    adapt_avg='Adaptation Index')

features_noAP = ['input_resistance', 'sag', 'capacitance', 'v_rest']  # tau
features_ap = ['f_i_curve_slope', 'ap_threshold',
               'ap_width', 'ap_peak_to_threshold',
               'ap_trough_to_threshold', 'ap_upstroke',
               'ap_updownstroke_ratio']  # ap_trough, ap_peak, hs_firing_rate, ap_height, hs_latency, ap_downstroke
features_isi = ['hs_adaptation', 'adapt_avg', 'hs_median_isi']
qc_cols = ['v_baseline',
       'bias_current', 'vm_for_sag', 'rheobase_stim_amp', 'hold', 'ra_pre',
       'ra_post', 'start', 'step', 'ra_est']
id_cols = ['experiment', 'cell', 'recording', 'id', 'date']
meta_cols = ['has_ap', 'strain', 'dob', 'age', 'slicetype']

muted = {name: 'rgba(' + str(a) + ', ' + str(b) + ', ' + str(c) + ')' for name, (a, b, c) \
    in zip(['blue', 'green', 'red', 'purple', 'yellow', 'cyan'], sns.color_palette("muted"))}

def cluster_heatmap(data, features, idx_color_mapping, exp_lut, legend=True):
    '''draw heatmap with hierachical clustering for all cells using Seaborn.'''
    df = pd.DataFrame(data, columns=features).T
    correlations_array = np.asarray(df)

    #row_linkage = hierarchy.linkage(distance.pdist(correlations_array), method='ward', metric='correlation')
    #col_linkage = hierarchy.linkage(distance.pdist(correlations_array.T), method='ward', metric='correlation')

    g = sns.clustermap(df, method='ward', metric='euclidean', center=0,
                       col_colors = idx_color_mapping, figsize=(20,15))
    _ = plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=28)
    _ = plt.setp(g.ax_heatmap.get_xticklabels(), rotation=0, fontsize=22)

    if legend:
        legend_patches = []
        for k, v in exp_lut.items():
            legend_patches.append(mpatches.Patch(color=v, label=k))

        plt.legend(handles=legend_patches, fontsize=18)
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

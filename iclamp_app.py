# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import plotly.graph_objs as go
import plotly.tools as tls
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
import io
import base64

from scipy.spatial import distance
from scipy.cluster import hierarchy
import matplotlib.patches as mpatches

from sklearn import preprocessing
from sklearn.decomposition import PCA

app = dash.Dash()

# load data
ANALYSIS_DIR = os.path.expanduser('~/Dropbox/Data/organoid/organoid_ephys/analysis/iclamp_2017-09/')
DATA_DIR = os.path.expanduser('~/Dropbox/Data/organoid/organoid_ephys/')
unique_isteps = pd.read_excel(os.path.join(ANALYSIS_DIR, 'data/unique_isteps_narrow.xlsx'))
plot_paths = pd.read_csv(os.path.join(ANALYSIS_DIR, 'data/plot_path.csv'))
cells_all = unique_isteps[[x for x in unique_isteps.columns if x != 'duplicates']]
cells_ap = cells_all[cells_all['has_ap'] == 'Yes'].reset_index(drop=True)
cells_ap = cells_ap[cells_ap['cm_est'] < 75].reset_index(drop=True)  # get rid of outlier
cells_noap = cells_all[cells_all['has_ap'] == 'No'].reset_index(drop=True)

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

cells_ap_features = cells_ap[features_noAP + features_ap]
cells_adapt_features = cells_ap[features_noAP + features_ap + ['hs_adaptation']].fillna(value={'hs_adaptation':0})
cells_adapt_arr = np.array(cells_adapt_features)

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

# run PCA

scaler = preprocessing.StandardScaler().fit(cells_adapt_arr)
cells_adapt_scaled = scaler.transform(cells_adapt_arr)

pca = PCA(n_components = None)
pca.fit(cells_adapt_scaled)
cells_adapt_pca = pca.transform(cells_adapt_scaled)

cells_adapt_pca_minmax = preprocessing.MinMaxScaler().fit_transform(cells_adapt_pca) * 0.8 + 0.1
text_labels = ['-'.join([x, y]) for x, y in zip(cells_ap['experiment'], cells_ap['recording'])]

experiments = cells_ap['experiment'].unique()
colors = sns.color_palette("Set2", len(experiments))
exp_lut = dict(zip(experiments, colors))
idx_color_mapping = cells_ap['experiment'].map(exp_lut)

muted = {name: 'rgba(' + str(a) + ', ' + str(b) + ', ' + str(c) + ')' for name, (a, b, c) \
    in zip(['blue', 'green', 'red', 'purple', 'yellow', 'cyan'], sns.color_palette("muted"))}

# generate heatmap
# TODO: use plotly interactive heatmap

#heatmap_path = os.path.join(DATA_DIR, 'analysis/iclamp_2017-09/figures/cell_feature_heatmap_legend.png')
#encoded_heatmap = base64.b64encode(open(heatmap_path, 'rb').read())
#decoded_heatmap = 'data:image/png;base64,{}'.format(encoded_heatmap.decode())

def cluster_heatmap():
    df = pd.DataFrame(cells_adapt_scaled, columns=feature_names.values()).T
    correlations_array = np.asarray(df)

    row_linkage = hierarchy.linkage(distance.pdist(correlations_array), method='average')
    col_linkage = hierarchy.linkage(distance.pdist(correlations_array.T), method='average')

    g = sns.clustermap(df, row_linkage=row_linkage, col_linkage=col_linkage, center=0,
                       col_colors = idx_color_mapping, figsize=(20,15))
    _ = plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=28)
    _ = plt.setp(g.ax_heatmap.get_xticklabels(), rotation=0, fontsize=22)

    legend_patches = []
    for k, v in exp_lut.items():
        legend_patches.append(mpatches.Patch(color=v, label=k))

    plt.legend(handles=legend_patches, fontsize=28)
    return g

g = cluster_heatmap()
buf = io.BytesIO()
g.savefig(buf, format='png')
buf.seek(0)
encoded_heatmap = base64.b64encode(buf.getvalue())
buf.close()
decoded_heatmap = 'data:image/png;base64,{}'.format(encoded_heatmap.decode())

feature_order_hclust = g.dendrogram_row.reordered_ind[::-1]


# app layout

app.layout = html.Div([
    html.H1('Organoid single cell electrophysiology', style={'text-align': 'center'}),
    html.Div([
        html.Div([
            html.H3('Select features to plot on PCA.')
        ], style={'width':'49%', 'display': 'inline-block'}),
        html.Div([
            html.H3('', id='cell_info', style={'text-align': 'left'})
        ], style={'width':'49%', 'display': 'inline-block'})
    ]),
    html.Div([
        html.Div([
            html.Div([
                dcc.Dropdown(
                    id='feature_pc',
                    options=[{'label': 'None ', 'value': None},
                             {'label': 'Experiment ', 'value': 'experiment'}] + \
                            [{'label': v + ' ', 'value': k} for k, v in feature_names.items()],
                    value=None
                )
            ], style={'width': '30%'}),
        ], style={'width': '60%', 'display': 'inline-block'}),
        html.Div([

            dcc.RadioItems(
                id='png_gif',
                options=[
                    {'label': 'PNG ', 'value': 'png'},
                    {'label': 'GIF ', 'value': 'gif'},
                ],
                value='gif',
                labelStyle={'display': 'inline-block'}
            )
        ], style={'width': '39%', 'display': 'inline-block'})
    ], style={'width': '100%', 'display': 'flex', 'flex-flow': 'row wrap', 'justify-content': 'center'}),

    html.Div([
        html.Div([
            dcc.Graph(id='cell_pca_3d', hoverData={'points': [{'hoverinfo': '2017_01_27_0009'}]},
                    style={'height': '100%'})
            ], style={'width': '45%', 'height': '100%', 'display': 'inline-block', 'padding': '0 20'}),
        html.Div([
            html.Img(id='fi_spike_phase', src='',
                    style={'height': '100%'})
            ], style={'width': '12%', 'height': '100%', 'display': 'inline-block'}),
        html.Div([
            html.Img(id='isteps', src='',
                    style={'height': '100%'})
            ], style={'width': '40%', 'height': '100%', 'display': 'inline-block'})
    ], style={'height': '400px', 'width': '100%',
                'display': 'flex', 'flex-flow': 'row wrap', 'justify-content': 'center'}),


    html.Div([
        html.Div([
            html.H3('Hierarchical clustering heatmap'),
            html.Img(id='cell_feature_heatmap', src=decoded_heatmap,
                    style={'height': '100%'})
            ], style={'width': '55%', 'height': '100%', 'display': 'inline-block'}),
        html.Div([
            dcc.Graph(id='cell_bar',
                    style={'height': '100%'})
            ], style={'width': '30%', 'height': '130%', 'display': 'inline-block'}),
        html.Div([], style={'width': '13%', 'height': '100%', 'display': 'inline-block'})

    ], style={'height': '400px', 'width': '100%',
                'display': 'flex', 'flex-flow': 'row wrap', 'justify-content': 'center'})
], style={'width': '1600px', 'margin':'auto'})


# callback functions

@app.callback(
    dash.dependencies.Output('cell_pca_3d', 'figure'),
    [dash.dependencies.Input('feature_pc', 'value')])
def update_cell_pca_3d(feature):
    if feature is None:
        color = ['rgba(' + str(a) + ', ' + str(c) + ', ' + str(b) + ')' \
               for a,b,c in cells_adapt_pca_minmax[:,:3]]
    elif feature == 'experiment':
        color = ['rgba(' + str(a-0.001) + ', ' + str(b-0.001) + ', ' + str(c-0.001) + ')' \
               for a,b,c in idx_color_mapping]
        print(color)
    else:
        i = list(feature_names.keys()).index(feature)
        color = cells_adapt_scaled[:,i]

    return {
        'data': [go.Scatter3d(
            x=-cells_adapt_pca[:,0],
            y=cells_adapt_pca[:,1],
            z=cells_adapt_pca[:,2],
            mode='markers',
            text=text_labels,
            hoverinfo=[x for x in cells_ap['recording']],
            marker=dict(
                size=8,
                # color=['rgb(' + ', '.join(list(map(str, x))) + ')' for x in idx_color_mapping],
                color=color,
                colorscale='RdYlBu',
                # color='rgb(' + ', '.join(list(map(str, idx_color_mapping[0]))) + ')',
                opacity=0.8
            )
        )],
        'layout': go.Layout(
            title="PCA",
            scene = dict(
                xaxis=dict(title="PC-1 (%0.0f%%)" % (pca.explained_variance_ratio_[0]*100),
                          titlefont=dict(size=25)),
                yaxis=dict(title="PC-2 (%0.0f%%)" % (pca.explained_variance_ratio_[1]*100),
                          titlefont=dict(size=25)),
                zaxis=dict(title="PC-3 (%0.0f%%)" % (pca.explained_variance_ratio_[2]*100),
                          titlefont=dict(size=25)),
                camera = dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=-1, y=2, z=0.7)
                )
            ),
            margin=dict(
                l=0,
                r=0,
                b=0,
                t=0
            ),
        )
    }

@app.callback(
    dash.dependencies.Output('cell_bar', 'figure'),
    [dash.dependencies.Input('cell_pca_3d', 'hoverData')])
def update_cell_bar(hoverData):
    recording = hoverData['points'][0]['hoverinfo']
    features_scaled = cells_adapt_scaled[cells_ap.recording == recording][0][feature_order_hclust]
    features_raw = cells_adapt_arr[cells_ap.recording == recording][0][feature_order_hclust]
    features_name = cells_adapt_features.columns[feature_order_hclust]

    background = [go.Scatter(
        x=cells_adapt_scaled[i][feature_order_hclust],
        y=features_name,
        #orientation = 'h',
        mode='markers+lines',
        line=dict(
            color='lightgrey',
            width=2),
        marker=dict(
            size=5,
            color='lightblue',
            line=dict(
                color='grey',
                width=1.5,
            )
        ),
        opacity=0.2
    ) for i in range(len(cells_adapt_features))]

    hightlight = go.Scatter(
        x=features_scaled,
        y=features_name,
        #orientation = 'h',
        mode='markers+lines+text',
        text=[str(x) for x in features_raw],
        textposition='right',
        line=dict(
            color=muted['green'],
            width=2),
        marker=dict(
            size=15,
            color=muted['blue'],
            line=dict(
                color='grey',
                width=1.5,
            )
        ),
        opacity=0.8
    )

    data = [*background, hightlight]
    layout = go.Layout(
        title=None,
        showlegend=False,
        xaxis=dict(
            showticklabels=False,
        ),
        yaxis=dict(
            showticklabels=False,
        ),
        margin=dict(
            l=00,
            r=150,
            b=55,
            t=105
        ),

    )
    return {'data': data, 'layout': layout}

@app.callback(
    dash.dependencies.Output('isteps', 'src'),
    [dash.dependencies.Input('cell_pca_3d', 'hoverData'),
    dash.dependencies.Input('png_gif', 'value')])
def update_istep_plots(hoverData, png_gif):
    recording = hoverData['points'][0]['hoverinfo']
    plot_paths_row = plot_paths[plot_paths.recording == recording]
    gif_path, png_path = \
        [os.path.join(DATA_DIR, plot_paths_row[x].item()) \
            for x in ['gif_path', 'png_path']]
    if png_gif == 'gif':
        encoded_img = base64.b64encode(open(gif_path, 'rb').read())
        decoded_img = 'data:image/gif;base64,{}'.format(encoded_img.decode())
    else:
        encoded_img = base64.b64encode(open(png_path, 'rb').read())
        decoded_img = 'data:image/o=png;base64,{}'.format(encoded_img.decode())

    return decoded_img

@app.callback(
    dash.dependencies.Output('fi_spike_phase', 'src'),
    [dash.dependencies.Input('cell_pca_3d', 'hoverData')])
def update_istep_plots(hoverData):

    recording = hoverData['points'][0]['hoverinfo']
    plot_paths_row = plot_paths[plot_paths.recording == recording]
    fi_spike_phase_path = os.path.join(DATA_DIR, plot_paths_row['fi_spike_phase_mid'].item())
    encoded_img = base64.b64encode(open(fi_spike_phase_path, 'rb').read())
    decoded_img = 'data:image/png;base64,{}'.format(encoded_img.decode())
    return decoded_img

@app.callback(
    dash.dependencies.Output('cell_info', 'children'),
    [dash.dependencies.Input('cell_pca_3d', 'hoverData')])
def update_cell_info(hoverData):
    # print(hoverData)
    recording = hoverData['points'][0]['hoverinfo']
    #print(recording)
    cell_info = cells_ap[cells_ap.recording == recording]
    idx = cell_info.index[0]
    #print(cell_info[['date', 'strain', 'cell', 'recording']])
    return 'Date: {} -- Strain: {}  --  Recording: {} -- Index: {}  ' \
        .format(list(cell_info['date'])[0].strftime("%Y-%m-%d"), *[list(cell_info[x])[0] for x in ['strain', 'recording']], idx)

#app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})

if __name__ == '__main__':
    app.run_server(port=1234, debug=True)

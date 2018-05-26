# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import plotly.graph_objs as go
import plotly.tools as tls
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
import argparse
#import path
import json
import random

from sklearn import preprocessing
from sklearn.decomposition import PCA

from app_utils import *

with open("config.txt", 'r') as f:
    file_paths = json.load(f)

parser = argparse.ArgumentParser(description='Viz for Patch Clamp Electrophysiology')
parser.add_argument('--data',
                default=os.path.expanduser(file_paths['data']),
                type=str, help='Directory for raw data')
parser.add_argument('--analysis',
                default=os.path.expanduser(file_paths['analysis']),
                type=str, help='Directory for analysis files')
args = parser.parse_args()

# load data
# use pre-computed features stored on local drive (fetched from Datajoint.)
unique_isteps = pd.read_excel(os.path.join(args.analysis, 'unique_isteps_narrow.xlsx'))
plot_paths = pd.read_csv(os.path.join(args.analysis, 'plot_path.csv'))
#TODO: integrate with datajoint, and fetch data directly from database

def iclamp_viz(unique_isteps=unique_isteps, plot_paths=plot_paths):
    app = dash.Dash()

    # subset cells with and without action potentials
    cells_all = unique_isteps[[x for x in unique_isteps.columns if x != 'duplicates']]
    cells_ap = cells_all[(cells_all['has_ap'] == 'Yes') & (cells_all['hs_firing_rate'] >0)].reset_index(drop=True)
    # cells_ap = cells_ap[cells_ap['cm_est'] < 75].reset_index(drop=True)  # no need to remove outlier
    # cells_noap = cells_all[cells_all['has_ap'] == 'No'].reset_index(drop=True)

    # select useful features
    # cells_ap_features = cells_ap[features_noAP + features_ap]
    cells_adapt_features = cells_ap[features_noAP + features_ap + ['adapt_avg']].fillna(value={'adapt_avg':0})
    cells_adapt_arr_raw = np.array(cells_adapt_features)

    # log transform
    cells_adapt_features['ap_width'] = np.log10(cells_adapt_features['ap_width'])
    cells_adapt_features['ap_upstroke'] = np.log10(cells_adapt_features['ap_upstroke'])

    cells_adapt_arr = np.array(cells_adapt_features)

    # run PCA
    scaler = preprocessing.StandardScaler().fit(cells_adapt_arr)
    cells_adapt_scaled = scaler.transform(cells_adapt_arr)
    pca = PCA(n_components = None)
    pca.fit(cells_adapt_scaled)
    cells_adapt_pca = pca.transform(cells_adapt_scaled)
    cells_adapt_pca_minmax = preprocessing.MinMaxScaler().fit_transform(cells_adapt_pca) * 0.95 + 0.025

    # labels and color maps
    text_labels = ['-'.join([x, y]) for x, y in zip(cells_ap['experiment'], cells_ap['recording'])]
    experiments = cells_ap['experiment'].unique()
    cluster_colors = sns.hls_palette(len(experiments), l=0.7)
    random.seed(0)
    colors = random.sample(cluster_colors, len(cluster_colors), )
    exp_lut = dict(zip(experiments, colors))
    idx_color_mapping = cells_ap['experiment'].map(exp_lut)

    # generate cluster heatmap using Seaborn
    features = [feature_names[x] for x in cells_adapt_features.columns]
    g = cluster_heatmap(cells_adapt_scaled, features, idx_color_mapping, exp_lut, legend=False)
    feature_order_hclust = g.dendrogram_row.reordered_ind[::-1]
    # encode the heatmap png image into memory buffer
    decoded_heatmap = byte_encode_img(g)
    # TODO: use plotly interactive heatmap, so that clicking on a cell highlights a heatmap column

    # app layout
    app.layout = html.Div([
        html.H1('Interactive Visualization for Patch Clamp Electrophysiology',
                style={'text-align': 'center', 'font-family':'helvetica', 'font-weight':'normal'}),
        html.Div([
            html.Div([
                html.H2('Select features to plot on PCA', id='plot_feature',
                    style={'text-align': 'center', 'font-family':'helvetica', 'font-weight':'normal'})
            ], style={'width':'42%', 'display': 'inline-block'}),
            html.Div([
                html.H3('', id='cell_info',
                    style={'text-align': 'left', 'font-family':'helvetica', 'font-weight':'normal'})
            ], style={'width':'57%', 'display': 'inline-block'})
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
            ], style={'width': '54%', 'display': 'inline-block'}),
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
            ], style={'width': '45%', 'display': 'inline-block'})
        ], style={'width': '100%', 'display': 'flex', 'flex-flow': 'row wrap', 'justify-content': 'center'}),

        html.Div([
            html.Div([
                dcc.Graph(id='cell_pca_3d', hoverData={'points': [{'hoverinfo': '2017_01_27_0009'}]},
                        style={'height': '100%'})
                ], style={'width': '42%', 'height': '100%', 'display': 'inline-block', 'padding': '0 20'}),
            html.Div([
                html.Img(id='fi_spike_phase', src='',
                        style={'height': '100%'})
                ], style={'width': '10%', 'height': '100%', 'display': 'inline-block'}),
            html.Div([
                html.Img(id='isteps', src='',
                        style={'height': '100%'})
                ], style={'width': '30%', 'height': '100%', 'display': 'inline-block'}),
            html.Div([
                dcc.Graph(id='cell_bar_2',
                        style={'height': '100%'})
                ], style={'width': '18%', 'height': '100%', 'display': 'inline-block'})

        ], style={'height': '400px', 'width': '100%',
                    'display': 'flex', 'flex-flow': 'row wrap', 'justify-content': 'center'}),


        html.Div([
            html.Div([
                html.H3('Hierarchical clustering heatmap', style={'font-family':'helvetica', 'font-weight':'normal'}),
                html.Img(id='cell_feature_heatmap', src=decoded_heatmap,
                        style={'height': '100%'})
                ], style={'width': '60%', 'height': '100%', 'display': 'inline-block'}),
            html.Div([
                dcc.Graph(id='cell_bar',
                        style={'height': '100%'})
                ], style={'width': '30%', 'height': '130%', 'display': 'inline-block'}),
            html.Div([], style={'width': '20%', 'height': '100%', 'display': 'inline-block'})

        ], style={'height': '400px', 'width': '100%',
                    'display': 'flex', 'flex-flow': 'row wrap', 'justify-content': 'center'})
    ], style={'width': '1600px', 'margin':'auto'})


    # callback functions

    @app.callback(
        dash.dependencies.Output('plot_feature', 'children'),
        [dash.dependencies.Input('feature_pc', 'value')])
    def return_feature(feature):
        '''update the feature name used for PCA'''
        if feature in feature_names:
            return feature_names[feature]
        elif feature is None:
            return 'Select features to plot on PCA'
        else:
            return feature.title()

    @app.callback(
        dash.dependencies.Output('cell_pca_3d', 'figure'),
        [dash.dependencies.Input('feature_pc', 'value')])
    def update_cell_pca_3d(feature):
        '''draw 3D PCA plot with selected feature'''
        if feature is None:
            color = ['rgba(' + str(1-a) + ', ' + str(b) + ', ' + str(c) + ')' \
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
                x=cells_adapt_pca[:,0],
                y=cells_adapt_pca[:,2],
                z=cells_adapt_pca[:,1],
                mode='markers',
                text=text_labels,
                hoverinfo=[x for x in cells_ap['recording']],
                marker=dict(
                    size=8,
                    # color=['rgb(' + ', '.join(list(map(str, x))) + ')' for x in idx_color_mapping],
                    color=color,
                    colorbar=go.ColorBar(len=0.25,showticklabels=False, thickness=15, outlinewidth=0),
                    colorscale='RdYlBu',
                    # color='rgb(' + ', '.join(list(map(str, idx_color_mapping[0]))) + ')',
                    opacity=0.8
                )
            )],
            'layout': go.Layout(
                title="PCA",
                scene = dict(
                    xaxis=dict(title="PC-1 (%0.0f%%)" % (pca.explained_variance_ratio_[0]*100),
                              titlefont=dict(size=22)),
                    yaxis=dict(title="PC-3 (%0.0f%%)" % (pca.explained_variance_ratio_[2]*100),
                              titlefont=dict(size=22)),
                    zaxis=dict(title="PC-2 (%0.0f%%)" % (pca.explained_variance_ratio_[1]*100),
                              titlefont=dict(size=22)),
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
        dash.dependencies.Output('isteps', 'src'),
        [dash.dependencies.Input('cell_pca_3d', 'hoverData'),
        dash.dependencies.Input('png_gif', 'value')])
    def update_istep_plots(hoverData, png_gif):
        '''show all traces of current injection and membrane voltage for a selected cell_pca_3d
        Inputs:
            png_gif: either static png or animated gif
        '''
        recording = hoverData['points'][0]['hoverinfo']
        plot_paths_row = plot_paths[plot_paths.recording == recording]
        gif_path, png_path = \
            [os.path.join(args.data, plot_paths_row[x].item()) \
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
    def update_istep_side_plots(hoverData):
        '''show F-I curve, first action potential and phase plane plot.'''
        recording = hoverData['points'][0]['hoverinfo']
        plot_paths_row = plot_paths[plot_paths.recording == recording]
        fi_spike_phase_path = os.path.join(args.data, plot_paths_row['fi_spike_phase_mid'].item())
        encoded_img = base64.b64encode(open(fi_spike_phase_path, 'rb').read())
        decoded_img = 'data:image/png;base64,{}'.format(encoded_img.decode())
        return decoded_img

    @app.callback(
        dash.dependencies.Output('cell_info', 'children'),
        [dash.dependencies.Input('cell_pca_3d', 'hoverData')])
    def update_cell_info(hoverData):
        '''print the metadata (rerording, date, age, etc) of the selected cell.'''
        # print(hoverData)
        recording = hoverData['points'][0]['hoverinfo']
        #print(recording)
        cell_info = cells_ap[cells_ap.recording == recording]
        idx = cell_info.index[0]
        #print(cell_info[['date', 'strain', 'cell', 'recording']])
        return 'Date: {} -- Strain: {}  --  Recording: {} -- Index: {}  ' \
            .format(list(cell_info['date'])[0].strftime("%Y-%m-%d"), *[list(cell_info[x])[0] for x in ['strain', 'recording']], idx)

    @app.callback(
        dash.dependencies.Output('cell_bar', 'figure'),
        [dash.dependencies.Input('cell_pca_3d', 'hoverData')])
    def update_cell_bar(hoverData):
        '''Scatter plot highlighting all features of a selected cell.'''
        recording = hoverData['points'][0]['hoverinfo']
        features_scaled = cells_adapt_scaled[cells_ap.recording == recording][0][feature_order_hclust]
        features_raw = cells_adapt_arr_raw[cells_ap.recording == recording][0][feature_order_hclust]
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
            opacity=0.2,
            hoverinfo='none'
        ) for i in range(len(cells_adapt_features))]

        hightlight = go.Scatter(
            x=features_scaled,
            y=features_name,
            #orientation = 'h',
            mode='markers+lines+text',
            text=[format(x, '.3g') for x in features_raw],
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
            opacity=0.8,
            hoverinfo='none'
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
        dash.dependencies.Output('cell_bar_2', 'figure'),
        [dash.dependencies.Input('cell_pca_3d', 'hoverData')])
    def update_cell_bar_2(hoverData):
        '''Scatter plot highlighting all features of a selected cell.'''
        recording = hoverData['points'][0]['hoverinfo']
        features_scaled = cells_adapt_scaled[cells_ap.recording == recording][0][feature_order_hclust]
        features_raw = cells_adapt_arr_raw[cells_ap.recording == recording][0][feature_order_hclust]
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
            opacity=0.2,
            hoverinfo='none'
        ) for i in range(len(cells_adapt_features))]

        hightlight = go.Scatter(
            x=features_scaled,
            y=features_name,
            #orientation = 'h',
            mode='markers+lines+text',
            text=[feature_names[y] for y in features_name],
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
            opacity=0.8,
            hoverinfo='none'
        )

        hightlight_2 = go.Scatter(
            x=features_scaled,
            y=features_name,
            #orientation = 'h',
            mode='markers+text',
            text=[format(x, '.3g') for x in features_raw],
            textposition='left',
            marker=dict(
                size=15,
                color=muted['blue'],
                opacity=0
            ),
            opacity=0.8,
            hoverinfo='none'
        )

        data = [*background, hightlight, hightlight_2]
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
                l=0,
                r=0,
                b=0,
                t=0
            ),

        )
        return {'data': data, 'layout': layout}

    #app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})
    return app

if __name__ == '__main__':
    app = iclamp_viz()
    app.run_server(host='0.0.0.0', port=1235, debug=True)

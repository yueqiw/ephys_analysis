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

import random

from .viz_utils import *
from .feature_annotations import *

def iclamp_viz(input_data, plot_paths, data_root_dir):
    app = dash.Dash()

    # subset cells with and without action potentials
    cells_ap = input_data.loc[(input_data['has_ap'] == 'Yes'),:].copy()

    # select useful features

    features_pca = features_noAP + features_ap # adapt_avg has missing data, cannot use for pca
    features_raw = features_noAP + features_ap + ['adapt_avg'] # NOTE: this has to be the same order features_cluster
    features_cluster = log_features_noAP + log_features_ap + ['adapt_avg']
    cells_features_raw = cells_ap[features_raw]
    cells_features_raw_scaled = (cells_features_raw - cells_features_raw.mean()) / cells_features_raw.std(ddof=0)

    # run PCA
    pca, cells_pca, cells_scaled = run_pca(cells_ap[features_pca])
    cells_pca_minmax = preprocessing.MinMaxScaler().fit_transform(cells_pca) * 0.95 + 0.025

    # TODO: add UMAP

    # labels and color maps
    text_labels = ['-'.join([x, y]) for x, y in zip(cells_ap['experiment'], cells_ap['recording'])]

    # generate cluster heatmap using Seaborn
    k_cluster = 5
    with sns.axes_style(None, {'axes.facecolor':'#e5e5e5'}):
        g, hclust_labels = cluster_heatmap(data_df=cells_ap[features_cluster], feature_name_dict=feature_name_dict,
                        categorical_df=cells_ap, categories=['cluster', 'strain'], k_cluster=k_cluster,
                        method='average', metric='correlation', row_cluster=True, mask=None,
                        caterogy_color_l=0.65, caterogy_color_s=[0.65, 0.4, 0.5], color_seed=1)

    cells_ap['cluster'] = hclust_labels
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
                        options=[{'label': 'None ', 'value': None}] + \
                                [{'label': k.capitalize() + ' ', 'value': k} for k in metadata_in_dropdown] + \
                                [{'label': v + ' ', 'value': k} for k, v in feature_name_dict.items()],
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
                # dcc.Graph(id='cell_bar',
                #         style={'height': '100%'})
                ], style={'width': '30%', 'height': '130%', 'display': 'inline-block'}),
            html.Div([], style={'width': '20%', 'height': '100%', 'display': 'inline-block'})

        ], style={'height': '600px', 'width': '100%',
                    'display': 'flex', 'flex-flow': 'row wrap', 'justify-content': 'center'})
    ], style={'width': '1600px', 'margin':'auto'})


    # callback functions

    @app.callback(
        dash.dependencies.Output('plot_feature', 'children'),
        [dash.dependencies.Input('feature_pc', 'value')])
    def return_feature(feature):
        '''update the feature name used for PCA'''
        if feature in feature_name_dict:
            return feature_name_dict[feature]
        elif feature is None:
            return 'Select features to plot on PCA'
        else:
            return feature.title()

    @app.callback(
        dash.dependencies.Output('cell_pca_3d', 'figure'),
        [dash.dependencies.Input('feature_pc', 'value')])
    def update_cell_pca_3d(feature):
        '''draw 3D PCA plot with selected feature'''
        # cells_pca_minmax
        # metadata_in_dropdown
        # cells_ap
        # cells_scaled
        # cells_pca
        # text_labels
        # pca
        if feature is None:
            color = ['rgba(' + str(1-a) + ', ' + str(b) + ', ' + str(c) + ')' \
                   for a,b,c in cells_pca_minmax[:,:3]]
        elif feature in metadata_in_dropdown:
            color = ['rgba(' + str(a-0.001) + ', ' + str(b-0.001) + ', ' + str(c-0.001) + ')' \
                   for a,b,c in categorical_color_mapping(cells_ap[feature])[0]]
            print(color)
        elif feature in features_pca:
            i = list(features_pca).index(feature)
            color = cells_scaled[:,i]

        return {
            'data': [go.Scatter3d(
                x=cells_pca[:,0],
                y=cells_pca[:,2],
                z=cells_pca[:,1],
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

        if png_gif == 'gif' and type(plot_paths_row['istep_gif_path'].item()) is str:
            gif_path = os.path.join(data_root_dir, plot_paths_row['istep_gif_path'].item())
            encoded_img = base64.b64encode(open(gif_path, 'rb').read())
            decoded_img = 'data:image/gif;base64,{}'.format(encoded_img.decode())
        else:
            png_path = os.path.join(data_root_dir, plot_paths_row['istep_png_mid_path'].item())
            encoded_img = base64.b64encode(open(png_path, 'rb').read())
            decoded_img = 'data:image/png;base64,{}'.format(encoded_img.decode())

        return decoded_img

    @app.callback(
        dash.dependencies.Output('fi_spike_phase', 'src'),
        [dash.dependencies.Input('cell_pca_3d', 'hoverData')])
    def update_istep_side_plots(hoverData):
        '''show F-I curve, first action potential and phase plane plot.'''
        recording = hoverData['points'][0]['hoverinfo']
        plot_paths_row = plot_paths[plot_paths.recording == recording]
        fi_spike_phase_path = os.path.join(data_root_dir, plot_paths_row['mid_fi_spike_phase'].item())
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
        dash.dependencies.Output('cell_bar_2', 'figure'),
        [dash.dependencies.Input('cell_pca_3d', 'hoverData')])
    def update_cell_bar_2(hoverData):
        '''Scatter plot highlighting all features of a selected cell.'''
        recording = hoverData['points'][0]['hoverinfo']
        cells_selected_scaled = cells_features_raw_scaled[cells_ap.recording == recording].values[0][feature_order_hclust]
        cells_selected_raw = cells_features_raw[cells_ap.recording == recording].values[0][feature_order_hclust]
        features_name = np.array(features_raw)[feature_order_hclust]

        background = [go.Scatter(
            x=cells_features_raw_scaled.iloc[i, :][feature_order_hclust],
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
        ) for i in range(len(cells_features_raw_scaled))]

        hightlight = go.Scatter(
            x=cells_selected_scaled,
            y=features_name,
            #orientation = 'h',
            mode='markers+lines+text',
            text=[feature_name_dict[y] for y in features_name],
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
            x=cells_selected_scaled,
            y=features_name,
            #orientation = 'h',
            mode='markers+text',
            text=[format(x, '.3g') for x in cells_selected_raw],
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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os, sys
from scipy.spatial import distance
from scipy.cluster import hierarchy
from scipy import stats

import matplotlib.patches as mpatches
import io
import base64
import random

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn import metrics, cluster
from sklearn import neighbors

import plotly.plotly as py
import plotly.graph_objs as go
import plotly

import umap, numba



#sns.set_style("white")

metadata_in_dropdown = ['experiment', 'strain', 'cluster']

muted = {name: 'rgba(' + str(a) + ', ' + str(b) + ', ' + str(c) + ')' for name, (a, b, c) \
    in zip(['blue', 'green', 'red', 'purple', 'yellow', 'cyan'], sns.color_palette("muted"))}

def cluster_heatmap(data_df, feature_name_dict, categorical_df, categories, k_cluster=None,
                    method='ward', metric='euclidean', row_method=None, row_metric=None,
                    row_cluster=True, pairwise_complete_obs=True,
                    n_pca=None, caterogy_color_l=0.65, caterogy_color_s=0.65, color_seed=0,
                    figsize=(30,15), fontsize=(24, 20, 15), plot_x_labels=True, legend=True, mask=None, **kwargs):
    '''draw heatmap with hierachical clustering for all cells using Seaborn.'''
    np.all(data_df.index == categorical_df.index)

    if row_method is None:
        row_method = method
    if row_metric is None:
        row_metric = metric

    if pairwise_complete_obs and metric == 'correlation':  # use pandas corr()
        data_scaled = (data_df - data_df.mean()) / data_df.std(ddof=0)
        dist_corr_obs = distance.squareform(1 - data_scaled.T.corr())  # pairwise complete (allow na)
        dist_corr_features = distance.squareform(1 - data_scaled.corr())  # pairwise complete (allow na)
        data_scaled = data_scaled.values

        row_linkage = hierarchy.linkage(dist_corr_features, method=row_method, metric=row_metric)
        col_linkage = hierarchy.linkage(dist_corr_obs, method=method, metric=metric)
    else:
        scaler = preprocessing.StandardScaler().fit(data_df)
        data_scaled = scaler.transform(data_df)
        data_scaled_obs = data_scaled
        dist_corr_features = data_scaled.T
        if not n_pca is None:
            pca = PCA(n_components = None)
            pca.fit(data_scaled)
            data_pca = pca.transform(data_scaled)
            print(pca.explained_variance_ratio_.cumsum())
            data_scaled_obs = data_pca[:, :n_pca]
        row_linkage = hierarchy.linkage(dist_corr_features, method=row_method, metric=row_metric)
        col_linkage = hierarchy.linkage(data_scaled_obs, method=method, metric=metric)

    scaled_df = pd.DataFrame(data_scaled,
                             columns=[feature_name_dict.get(x, x) for x in data_df.columns],
                             index=data_df.index).T

    categorical_all = []
    hclust_labels = None
    for feature in categories:
        if feature == 'cluster':
            if k_cluster is None:
                raise ValueError('k_cluster not provided.')
            hclust_labels = hierarchy.fcluster(col_linkage, k_cluster, criterion='maxclust')
            hclust_labels = pd.Series(hclust_labels, name='cluster', index=data_df.index)
            categorical_all.append(hclust_labels)
        else:
            categorical_all.append(categorical_df[feature])

    if isinstance(caterogy_color_l, float):
        caterogy_color_l = [caterogy_color_l] * len(categorical_all)
    if isinstance(caterogy_color_s, float):
        caterogy_color_s = [caterogy_color_s] * len(categorical_all)
    # this may break if the inputs are not float or lists of the correct length
    cat, luts = list(zip(*[categorical_color_mapping(x, l=l, s=s, seed=color_seed)
                           for x, l, s in zip(categorical_all, caterogy_color_l, caterogy_color_s)]))

    g = sns.clustermap(scaled_df, center=0, row_linkage=row_linkage, col_linkage=col_linkage,
                        row_cluster=row_cluster, mask=mask,
                       col_colors = pd.DataFrame(list(cat)).T, figsize=figsize, cmap='RdBu_r', **kwargs)
    _ = plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=fontsize[0])
    _ = plt.setp(g.ax_col_colors.get_yticklabels(), rotation=0, fontsize=fontsize[1])
    g.ax_heatmap.set_xlabel("")

    if plot_x_labels:
        _ = plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90, fontsize = fontsize[2])
        last_anchor = -0.25
    else:
        g.ax_heatmap.set_xticklabels([])
        last_anchor = -0.05

    if legend:
        for i, (lut, legend_name) in enumerate(zip(luts, categories)):
            legend_patches = []
            legend_list = [(k, v) for k, v in lut.items()]
            legend_list = sorted(legend_list, key=lambda x: x[0])
            for k, v in legend_list:
                legend_patches.append(mpatches.Patch(color=v, label=k))
            avg_legend_len = np.mean([len(x) if isinstance(x, str) else 1 for x in lut.keys()])
            ncol = int(figsize[0] * 3 / max(avg_legend_len, 10))
            last_anchor -= 0.05 * (int((len(lut)-1)/ncol)+1+1)
            if i != len(luts) - 1:
                legend = g.ax_heatmap.legend(handles=legend_patches, fontsize=20, loc='lower center', bbox_to_anchor=[0.5, last_anchor], ncol=ncol, title=legend_name.title())
                legend.get_title().set_fontsize(24)
                legend = g.ax_heatmap.add_artist(legend)
            else:
                legend = g.ax_heatmap.legend(handles=legend_patches, fontsize=20, loc='lower center', bbox_to_anchor=[0.5, last_anchor], ncol=ncol, title=legend_name.title())
                legend.get_title().set_fontsize(24)
    return g, hclust_labels


def run_pca(data_df):
    scaler = preprocessing.StandardScaler().fit(data_df)
    data_scaled = scaler.transform(data_df)
    pca = PCA(n_components = None)
    pca.fit(data_scaled)
    data_pca = pca.transform(data_scaled)
    return pca, data_pca, data_scaled

def categorical_color_mapping(data, l=0.7, s=0.7, seed=0):
    categories = np.unique(data)
    colors = sns.hls_palette(len(categories), l=l, s=s)
    random.seed(seed)
    colors = random.sample(colors, len(colors), )
    lut = dict(zip(categories, colors))
    cat_color_mapping = data.map(lut)
    return cat_color_mapping, lut

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

def silhouette_plot(X_data, k_range=[5], method='ward', metric='euclidean', pairwise_complete_obs=True, n_pca=None):
    if metric == 'precomputed':
        dist = distance.squareform(X_data)
    else:
        if pairwise_complete_obs and metric == 'correlation':  # use pandas corr()
            data_scaled = (X_data - X_data.mean()) / X_data.std(ddof=0)
            dist = distance.squareform(1 - data_scaled.T.corr())  # pairwise complete (allow na)
        else:
            scaler = preprocessing.StandardScaler().fit(X_data)
            data_scaled = scaler.transform(X_data)

    if not method == 'kmeans':
        if metric == 'precomputed' or (pairwise_complete_obs and metric == 'correlation'):
            linkage = hierarchy.linkage(dist, method=method, metric=metric)
        else:
            data_scaled_obs = data_scaled
            if not n_pca is None:
                pca = PCA(n_components = None)
                pca.fit(data_scaled)
                data_pca = pca.transform(data_scaled)
                print(pca.explained_variance_ratio_.cumsum())
                data_scaled_obs = data_pca[:, :n_pca]
            linkage = hierarchy.linkage(data_scaled_obs, method=method, metric=metric)

    fig, axes = plt.subplots(1, len(k_range), figsize=(5*len(k_range),5))
    for i, n_clusters in enumerate(k_range):
        ax = axes[i]
        if method == 'kmeans':
            clusterer = cluster.KMeans(n_clusters=n_clusters, random_state=0)
            cluster_labels = clusterer.fit_predict(data_scaled)
        else:
            cluster_labels = hierarchy.fcluster(linkage, n_clusters, criterion='maxclust')
        cluster_labels = pd.Series(cluster_labels, name='cluster', index=X_data.index)

        ax.set_xlim([-1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax.set_ylim([0, len(X_data) + (n_clusters + 1) * 2])

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters

        if metric == 'precomputed' or (pairwise_complete_obs and metric == 'correlation'):
            silhouette_data = distance.squareform(dist)
        else:
            silhouette_data = data_scaled
        silhouette_avg = metrics.silhouette_score(silhouette_data, cluster_labels, metric=metric)
        sample_silhouette_values = metrics.silhouette_samples(silhouette_data, cluster_labels, metric=metric)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)

        # Compute the silhouette scores for each sample
        _, lut = categorical_color_mapping(cluster_labels)

        y_lower = 2
        for i in sorted(list(lut.keys())):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = lut[i]

            ax.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 2  # 10 for the 0 samples

        ax.set_title("The silhouette plot for the clusters.")
        ax.set_xlabel("The silhouette coefficient values")
        ax.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax.set_yticks([])  # Clear the yaxis labels / ticks
    return fig


def plotly_3d_pca(data_df, categorical_df, color_labels, text_labels=None):
    pca, data_pca, data_scaled = run_pca(data_df)
    data_pca_minmax = preprocessing.MinMaxScaler().fit_transform(data_pca) * 0.95 + 0.025
    if color_labels is None:
        color = cells_adapt_pca_minmax[:,:3]
    else:
        color = categorical_color_mapping(color_labels)[0]
    hover_labels = ['-'.join([x, y]) for x, y in zip(categorical_df['experiment'], categorical_df['recording'])]
    if text_labels is None:
        mode = 'markers'
    else:
        mode = 'markers+text'

    trace1 = go.Scatter3d(
        x=data_pca[:,0],
        y=data_pca[:,2],
        z=data_pca[:,1],
        mode=mode,
        text=text_labels,
        hovertext=hover_labels,
        marker=dict(
            size=8,
            # color=['rgb(' + ', '.join(list(map(str, x))) + ')' for x in idx_color_mapping],
            color=['rgba(' + str(a) + ', ' + str(b) + ', ' + str(c) + ')' \
                   for a,b,c in color],
            # color='rgb(' + ', '.join(list(map(str, idx_color_mapping[0]))) + ')',
            opacity=0.8
        )
    )

    data = [trace1]
    layout = go.Layout(

        title="PCA",
        scene = dict(
            xaxis=dict(title="PC-1 (%0.0f%%)" % (pca.explained_variance_ratio_[0]*100),
                      titlefont=dict(size=25)),
            yaxis=dict(title="PC-3 (%0.0f%%)" % (pca.explained_variance_ratio_[2]*100),
                      titlefont=dict(size=25)),
            zaxis=dict(title="PC-2 (%0.0f%%)" % (pca.explained_variance_ratio_[1]*100),
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

    output = {"data": data, "layout": layout}
    return output


@numba.njit()
def corr_dist_numba(arr1, arr2, na_marker=1e5):
    if na_marker:
        arr1_mask = np.invert(arr1>na_marker)
        arr2_mask = np.invert(arr2>na_marker)
        mask = np.logical_and(arr1_mask, arr2_mask)
        x = arr1[mask]
        y = arr2[mask]
    else:
        x = arr1
        y = arr2
    n = len(x)
    mx = x.mean()
    my = y.mean()
    xm, ym = x-mx, y-my
    r_num = np.sum(xm * ym)
    r_den = np.sqrt(np.sum(xm*xm) * np.sum(ym*ym))
    r = r_num / r_den
    return 1 - r


def compute_umap(data, has_na=True, n_neighbors=15, n_components=2, min_dist=0.1, spread=1.0,
                 metric="euclidean", alpha=1.0, local_connectivity=1.0, bandwidth=1.0,
                 random_state=0, init='spectral', copy=False, umap_kwargs={}):
    #scaler = preprocessing.StandardScaler().fit(data)
    #data_scaled = scaler.transform(data)
    mean = data.mean()
    std = data.std(ddof=0)
    data_scaled = (data - mean) / std
    data_scaled = data_scaled.values

    data_scaled[np.isnan(data_scaled)] = 1e6

    # params for umap-learn
    params_umap = {'n_neighbors': n_neighbors,
                   'n_components': n_components,
                   'min_dist': min_dist,
                   'spread': spread,
                   'metric': metric,
                   'alpha': alpha,
                   'init': init,
                   'local_connectivity': local_connectivity,
                   'bandwidth': bandwidth,
                   'random_state': random_state,
                   'verbose': 0,
                   **umap_kwargs
                   }

    um = umap.UMAP(**params_umap)
    data_umap = um.fit_transform(data_scaled)
    return data_umap, um, (mean, std)

def plot_feature_comparison(data, features_plot, x="cluster", hue="strain", cluster_subset=None, cluster_name='cluster',
                            feature_names=None, bar=True, swamp=True, palette='muted', subfigsize=(4.5,4)):
    if not cluster_subset is None:
        data = data.loc[np.isin(data[cluster_name], cluster_subset),:].copy()
    if feature_names is None:
        feature_names = dict()
    n_features = len(features_plot)
    nrow = int((n_features - 1) / 4) + 1
    ncol = 4 if n_features > 4 else n_features
    fig, axes = plt.subplots(nrow, ncol, figsize=(subfigsize[0] * ncol, subfigsize[1] * nrow))
    axes = axes.ravel()
    for i, feature in enumerate(features_plot):
        if bar is True:
            sns.barplot(x=x, y=feature, hue=hue, data=data, ci=68, capsize=0.2, errwidth=2, errcolor="dimgray",
                        palette=palette, saturation=0.7, ax=axes[i], dodge=True, alpha=.7)
            if swamp is True:
                sns.swarmplot(x=x, y=feature, hue=hue, data=data, palette=['0.5', '0.5'], ax=axes[i], dodge=True)
        elif swamp is True:
            sns.swarmplot(x=x, y=feature, hue=hue, data=data, palette=palette, ax=axes[i], dodge=True)
        else:
            raise ValueError("specify a plot type.")
        axes[i].set_ylabel(feature_names.get(feature, feature), size=15)
        axes[i].set_title(feature_names.get(feature, feature), size=15)
        axes[i].set_xticklabels(axes[i].get_xticklabels(), size=15, rotation = 0, ha="right")
        axes[i].set_xlabel(axes[i].get_xlabel(), size=15)
        if (i+1) % ncol == 0:
            axes[i].legend(loc='right', bbox_to_anchor=(1.4, 0.5))
        else:
            legend = axes[i].legend()
            legend.remove()
        if np.all(data.loc[:,feature] >= 0):
            axes[i].set_ylim(0, data.loc[:,feature].max() * 1.1)
    plt.tight_layout(w_pad=4)
    sns.despine()
    return fig

def corr_dist(arr1, arr2, na_marker=1e5):
    arr1 = arr1.astype(float)
    arr2 = arr2.astype(float)
    if na_marker:
        arr1[arr1>na_marker] = np.nan
        arr2[arr2>na_marker] = np.nan
    s1 = pd.Series(arr1)
    s2 = pd.Series(arr2)
    return 1 - s1.corr(s2)

def find_nearest_cluster(cells_ref, cells_query, ref_labels, metric, k=4, set_na_marker=1e6):
    cells_ref_scaled = (cells_ref - cells_ref.mean()) / cells_ref.std(ddof=0)
    cells_query_scaled = (cells_query - cells_ref.mean()) / cells_ref.std(ddof=0)
    if set_na_marker:
        cells_ref_scaled = cells_ref_scaled.fillna(1e6)
        cells_query_scaled = cells_query_scaled.fillna(1e6)
    neigh = neighbors.KNeighborsClassifier(n_neighbors=k, metric=metric, weights='distance')
    neigh.fit(cells_ref_scaled, ref_labels)
    nearest_clusters = neigh.predict(cells_query_scaled)
    nearest_clusters = pd.Series(nearest_clusters, index=cells_query_scaled.index)
    return nearest_clusters, neigh


def feature_ttest_by_cluster(data, features, group_1, group_2, cluster_subset=None, feature_name='strain', cluster_name='cluster'):
    pval_df = []
    for feature in features:
        pvals = []
        clusters_all = np.sort(data[cluster_name].unique())
        for cls in clusters_all:
            data_subset = data.loc[data[cluster_name] == cls,:]
            res = stats.ttest_ind(data_subset.loc[data_subset[feature_name]==group_1, feature],
                            data_subset.loc[data_subset[feature_name]==group_2, feature], equal_var = False)
            pvals.append(res.pvalue)
        pvals = pd.Series(pvals, index=clusters_all)
        pval_df.append(pvals)
    pval_df = pd.DataFrame(pval_df, index=features).T
    pval_df['cluster'] = pval_df.index
    for cls in clusters_all:
        data_subset = data.loc[data[cluster_name] == cls,:]
    pval_df['n_' + group_1] = data.loc[data[feature_name] == group_1, cluster_name].value_counts().reindex(clusters_all).fillna(0).astype(int)
    pval_df['n_' + group_2] = data.loc[data[feature_name] == group_2, cluster_name].value_counts().reindex(clusters_all).fillna(0).astype(int)
    return pval_df

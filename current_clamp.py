import os
import numpy as np
import pandas as pd
import six
from collections import OrderedDict

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from matplotlib import gridspec, animation
from PIL import Image, ImageDraw, ImageFont

import allensdk_0_14_2.ephys_features as ft

# TODO: implement plotting functions under a class object 

def load_current_step_add_itrace(abf_file, ihold, istart, istep, startend=None, filetype='abf', channels=[0]):
    '''
    Load current clamp recordings from pClamp .abf files with only voltage traces
    '''
    ch0 = channels[0]
    rec = stfio.read(abf_file)
    assert(rec[ch0].yunits == 'mV')

    data = OrderedDict()
    data['file_id'] = os.path.basename(abf_file).strip('.' + filetype)
    data['file_directory'] = os.path.dirname(abf_file)
    data['record_date'] = rec.datetime.date()
    data['record_time'] = rec.datetime.time()

    data['dt'] = rec.dt / 1000
    data['hz'] = 1./rec.dt * 1000
    data['time_unit'] = 's'

    data['n_channels'] = len(rec)
    data['channel_names'] = [rec[ch0].name, 'Current_simulated']
    data['channel_units'] = [rec[ch0].yunits, 'pA']
    data['n_sweeps'] = len(rec[ch0])
    data['sweep_length'] = len(rec[ch0][0])

    data['t'] = np.arange(0, data['sweep_length']) * data['dt']

    start_idx = ft.find_time_index(data['t'], startend[0])
    end_idx = ft.find_time_index(data['t'], startend[1])
    current = [np.zeros_like(data['t']) + ihold for i in range(data['n_sweeps'])]
    for i in range(data['n_sweeps']):
        current[i][start_idx:end_idx] += istart + istep * i

    data['voltage'] = rec[ch0]
    data['voltage'] = [x.asarray() for x in data['voltage']]
    data['current'] = current

    current_channel = stfio.Channel([stfio.Section(x) for x in current])
    current_channel.yunits = 'pA'
    current_channel.name = 'Current_simulated'
    chlist = [rec[ch0], current_channel]
    rec_with_current = stfio.Recording(chlist)
    rec_with_current.dt = rec.dt
    rec_with_current.xunits = rec.xunits
    rec_with_current.datetime = rec.datetime
    return rec_with_current, data


def plot_current_step(data, fig_height=6, x_scale=3.5, xlim=[0.3,3.2],
                        startend=None, offset=[0.2, 0.4], lw_scale=1, alpha_scale=1,
                        plot_gray_sweeps=True,
                        blue_sweep=None, rheobase_sweep=None, sag_sweeps=[], vlim=[-145,60], ilim=[-95,150],
                        spikes_sweep_id = None, spikes_t = None,
                        other_features=None, trough_name = 'spikes_trough_5w',
                        bias_current = 0.0,
                        highlight = 'deepskyblue',
                        highlight_rheobase=sns.color_palette("muted").as_hex()[2],
                        highlight_sag=sns.color_palette("muted").as_hex()[4],
                        skip_sweep=1, skip_point=10, save=False,
                        rasterized=True):
    '''
    Plot overlayed sweeps in current clamp protocol, with one sweep in blue color
    If detected spikes are provided, also plot detected spikes.
    '''

    plt.style.use('ggplot')

    fig_width = fig_height
    if (spikes_sweep_id is not None) and (spikes_t is not None):
        fig_height *= 4.0/3.0
        n_plots = 3
        height_ratios = [1,3,1]
    else:
        n_plots = 2
        height_ratios = [3,1]

    if startend is not None:
        assert(type(startend) is list and len(startend) == 2)
        start = startend[0] - offset[0]
        end = startend[1] + offset[1]
        xlim = [start, end]
        length = end - start
        figsize = (length * x_scale * fig_width / 6., fig_height)
    else:
        figsize = (fig_width, fig_height)

    fig = plt.figure(figsize=figsize)
    fig.patch.set_alpha(0)
    gs = gridspec.GridSpec(n_plots, 1, height_ratios=height_ratios)

    axes = [plt.subplot(gs[x]) for x in range(n_plots)]

    indices = [x for x in range(data['n_sweeps']) if x % skip_sweep ==0 or x == data['n_sweeps']-1]
    # print(indices)

    if blue_sweep is not None:
        assert(isinstance(blue_sweep, (int, np.integer)))
        if not blue_sweep in indices:
            indices.append(blue_sweep)
    else:
        blue_sweep = indices[-2]

    if rheobase_sweep is not None:
        assert(isinstance(rheobase_sweep, (int, np.integer)))
        if not rheobase_sweep in indices:
            indices.append(rheobase_sweep)

    for i in indices[::-1]:
        if i == rheobase_sweep:
            color = highlight_rheobase
            lw=1.25 * lw_scale
            size=8 * lw_scale
            alpha=1 * alpha_scale
        elif i == blue_sweep or i == data['n_sweeps'] + blue_sweep:
            color = highlight
            lw=1.25 * lw_scale
            size=8 * lw_scale
            alpha=1 * alpha_scale
        elif i in sag_sweeps:
            color = highlight_sag
            lw=1 * lw_scale
            size=8 * lw_scale
            alpha=1 * alpha_scale
        else:
            color = 'gray'
            lw=0.2 * lw_scale
            size=3 * lw_scale
            alpha=0.6 * alpha_scale
            if not plot_gray_sweeps:
                continue

        axes[-2].plot(data['t'][::skip_point], data['voltage'][i][::skip_point],
                        color=color, lw=lw, alpha=alpha, rasterized=rasterized)
        if i == rheobase_sweep and not other_features is None:
            threshold_t = other_features['spikes_threshold_t'][spikes_sweep_id==i]
            threshold_v = other_features['spikes_threshold_v'][spikes_sweep_id==i]
            trough_t = other_features[trough_name + '_t'][spikes_sweep_id==i]
            trough_v = other_features[trough_name + '_v'][spikes_sweep_id==i]
            axes[-2].scatter(threshold_t, threshold_v, marker='_', s=30, lw=1, c="black", alpha=alpha)
            axes[-2].scatter(trough_t, trough_v, marker='+', s=30, lw=1, c="black", alpha=alpha)

        axes[-1].plot(data['t'][::skip_point], data['current'][i][::skip_point] - bias_current,
                        color=color, lw=lw, alpha=alpha, rasterized=rasterized)

        if n_plots == 3:
            spikes = spikes_t[spikes_sweep_id==i]
            axes[0].scatter(spikes, np.ones_like(spikes) * i, marker='o', s=size, c=color, alpha=alpha)


    axes[-2].set_ylim(vlim)
    axes[-2].set_ylabel('Membrane Voltage (mV)', fontsize=16)
    axes[-2].set_xticklabels([])
    axes[-1].set_ylim(ilim)
    axes[-1].set_ylabel('Current (pA)', fontsize=16)
    axes[-1].set_xlabel('Time (s)', fontsize=16)
    axes[0].set_ylim([1, data['n_sweeps']])
    axes[0].set_xticklabels([])
    axes[0].set_ylabel('Sweeps', fontsize=16)

    for ax in axes:
        ax.set_xlim(xlim)
        ax.yaxis.set_label_coords(-0.64/figsize[0],0.5)
        ax.patch.set_alpha(0)
        ax.grid(False)
        for loc in ['top', 'right', 'bottom', 'left']:
            ax.spines[loc].set_visible(False)

    plt.tight_layout()
    if save is True:
        plt.savefig(os.path.join(data['file_directory'], data['file_id']) + '.png', dpi=300)
        #plt.savefig(os.path.join(data['file_directory'], data['file_id']) + '.svg')
        plt.savefig(os.path.join(data['file_directory'], data['file_id']) + '.pdf', dpi=300)

    return fig


def animate_current_step(data, fig_height=6, x_scale=3.5, xlim=[0.3,3.2],
                        startend=None, offset=[0.2, 0.4],
                        vlim=[-145,60], ilim=[-95,150],
                        spikes_sweep_id = None, spikes_t = None,
                        bias_current = 0.0, highlight = 'deepskyblue',
                        skip_point=10, save=False, save_filepath=None, fps=2.5, dpi=100, blit=True):
    '''
    Make animated GIF containing all the sweeps in current clamp protocol.
    If detected spikes are provided, also plot detected spikes.

    Note the slow speed of this function is due to image -> gif/mp4 conversion, not creating animation.
    Creating animation takes < 1s. Saving it takes 5 - 10s.
    '''
    fig_width = fig_height
    if (spikes_sweep_id is not None) and (spikes_t is not None):
        fig_height *= 4.0/3.0
        n_plots = 3
        height_ratios = [1,3,1]
    else:
        n_plots = 2
        height_ratios = [3,1]

    plt.style.use('ggplot')
    if startend is not None:
        assert(type(startend) is list and len(startend) == 2)
        start = startend[0] - offset[0]
        end = startend[1] + offset[1]
        xlim = [start, end]
        length = end - start
        figsize = (length * x_scale * fig_width / 6., fig_height)
    else:
        figsize = (fig_width, fig_height)

    fig = plt.figure(figsize=figsize)

    gs = gridspec.GridSpec(n_plots, 1, height_ratios=height_ratios)
    axes = [plt.subplot(gs[x]) for x in range(n_plots)]

    # plot background traces in light gray
    color = 'gray'
    lw=0.2
    size=2
    alpha=0.6
    for i in range(data['n_sweeps']):
        axes[-2].plot(data['t'][::skip_point], data['voltage'][i][::skip_point], color=color, lw=lw, alpha=alpha)
        axes[-1].plot(data['t'][::skip_point], data['current'][i][::skip_point] - bias_current, color=color, lw=lw, alpha=alpha)
        if n_plots == 3:
             spikes = spikes_t[spikes_sweep_id==i]
             axes[0].plot(spikes, np.ones_like(spikes) * i, marker='o', markersize=size, ls='', color=color, alpha=alpha)

    axes[-2].set_ylim(vlim)
    axes[-2].set_ylabel('Membrane Voltage (mV)', fontsize=16)
    axes[-2].set_xticklabels([])
    axes[-1].set_ylim(ilim)
    axes[-1].set_ylabel('Current (pA)', fontsize=16)
    axes[-1].set_xlabel('Time (s)', fontsize=16)
    if n_plots == 3:
        axes[0].set_ylim([1, data['n_sweeps']])
        axes[0].set_xticklabels([])
        axes[0].set_ylabel('Sweeps', fontsize=16)
    for ax in axes:
        ax.set_xlim(xlim)
        ax.yaxis.set_label_coords(-0.64/figsize[0],0.5)
        ax.patch.set_alpha(0)

    plt.tight_layout()

    # initialize plots of highlighted traces
    color = highlight
    lw=1.5
    size=3
    alpha=1
    plot_2, = axes[-2].plot([], [], color=color, lw=lw, alpha=alpha)
    plot_1, = axes[-1].plot([], [], color=color, lw=lw, alpha=alpha)
    if n_plots == 3:
        plot_0, = axes[0].plot([], [], marker='o', markersize=size, ls='', color=color, alpha=alpha)

    def init_animation():
        return plot_0, plot_1, plot_2

    # animate the highlighted traces
    def animate(j):
        plot_2.set_data(data['t'][::skip_point], data['voltage'][j][::skip_point])
        plot_1.set_data(data['t'][::skip_point], data['current'][j][::skip_point] - bias_current)
        if n_plots == 3:
            spikes = spikes_t[spikes_sweep_id==j]
            plot_0.set_data(spikes, np.ones_like(spikes) * j)
        return plot_0, plot_1, plot_2

    anim = animation.FuncAnimation(fig, animate, init_func=init_animation, frames=data['n_sweeps'], blit=blit)
    if save:
        if save_filepath is None:
            raise ValueError("Please provide save path (gif or mp4).")
        elif save_filepath.endswith('.gif'):
            # use default dpi=100. Setting other dpi values will produce wierd-looking plots.
            anim.save(save_filepath, writer='imagemagick', fps=fps, dpi=dpi)
        elif save_filepath.endswith('.mp4'):
            anim.save(save_filepath, writer='ffmpeg', fps=fps, dpi=dpi)

    return fig, anim


def plot_fi_curve(stim_amp, firing_rate, figsize=(4,4), save_filepath = None, color=sns.color_palette("muted").as_hex()[0]):
    '''
    Plot F-I curve
    '''
    mpl.rcParams.update(mpl.rcParamsDefault)
    fig, ax = plt.subplots(1,1,figsize=figsize)
    ax.plot(stim_amp, firing_rate, marker='o', linewidth=1.5, markersize=8, color=color)
    fig.gca().spines['right'].set_visible(False)
    fig.gca().spines['top'].set_visible(False)
    ax.set_ylabel('Spikes per second', fontsize=18)
    ax.set_xlabel('Current (pA)', fontsize=18)
    ax.yaxis.set_label_coords(-0.22 * 4 / figsize[0],0.5)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 14)
    ax.patch.set_alpha(0)
    fig.tight_layout()
    if save_filepath is not None:
        fig.savefig(save_filepath, dpi=200)
    return fig

def plot_vi_curve(stim_amp, voltage, figsize=(4,4), save_filepath = None, color="gray"):
    '''
    Plot V-I curve
    '''
    mpl.rcParams.update(mpl.rcParamsDefault)
    fig, ax = plt.subplots(1,1,figsize=figsize)
    ax.plot(stim_amp, voltage, marker='o', linewidth=1.5, markersize=8, color=color)
    fig.gca().spines['right'].set_visible(False)
    fig.gca().spines['top'].set_visible(False)
    ax.set_ylabel('Voltage (mV)', fontsize=18)
    ax.set_xlabel('Current (pA)', fontsize=18)
    ax.yaxis.set_label_coords(-0.22 * 4 / figsize[0],0.5)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 14)
    ax.patch.set_alpha(0)
    fig.tight_layout()
    if save_filepath is not None:
        fig.savefig(save_filepath, dpi=200)
    return fig


def plot_first_spike(data, features, time_zero='threshold', figsize=(4,4), lw_scale=1,
                    window=None, vlim=[-80, 60], color=sns.color_palette("muted").as_hex()[2],
                    other_markers = dict(),
                    save_filepath = None, rasterized=False):
    '''
    Plot the first action potential. Time window is something like:
    Inputs
    -----
    data: raw data of sweeps loaded by load_current_step()
    features: dictionary from extract_istep_features()
    time_zero: whether to use threshold or peak time
    window: time range in ms. such as [t-10, t+40] ms

    Returns
    -------
    figure object
    '''
    assert(time_zero in ['threshold', 'peak'])
    if time_zero == 'threshold':
        t0 = features['spikes_threshold_t'][0]
        if window is None:
            window = [-10, 40]
    elif time_zero == 'peak':
        t0 = features['spikes_peak_t'][0]
        if window is None:
            window = [-15, 35]

    ap_window = [t0 + x * 0.001 for x in window]
    start, end = [ft.find_time_index(data['t'], x) for x in ap_window]
    t = (data['t'][start:end] - data['t'][start]) * 1000 + window[0]
    v = data['voltage'][features['rheobase_index']][start:end]

    mpl.rcParams.update(mpl.rcParamsDefault)
    fig, ax = plt.subplots(1,1,figsize=figsize)
    ax.plot(t, v, color=color, lw=2 * lw_scale, rasterized=rasterized)

    threshold_time = (features['spikes_threshold_t'][0] - t0) * 1000
    ax.hlines(features['ap_threshold'], window[0], threshold_time,
                linestyles='dotted', color='grey')

    for i, (feature, col) in enumerate(other_markers.items()):
        feature_t = (features[feature + '_t'][0] - t0) * 1000
        feature_v = features[feature + '_v'][0]
        if feature_t > window[1]:
            continue
        ax.scatter(feature_t, vlim[0] + i*2 + 2, marker='+', s=50, lw=1.5, c=col)


    ax.set_ylim(vlim)
    fig.gca().spines['right'].set_visible(False)
    fig.gca().spines['top'].set_visible(False)
    ax.set_ylabel('Voltage (mV)', fontsize=18)
    ax.set_xlabel('Time (ms)', fontsize=18)
    ax.yaxis.set_label_coords(-0.22 * 4 / figsize[0],0.5)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 14)
    ax.patch.set_alpha(0)
    ax.grid(False)
    for loc in ['top', 'right', 'bottom', 'left']:
        ax.spines[loc].set_visible(False)
    fig.tight_layout()

    if save_filepath is not None:
        fig.savefig(save_filepath, dpi=200)
    return fig

def plot_phase_plane(data, features, filter=None, figsize=(4, 4), window=[-50, 200], lw_scale=1,
                        vlim=[-80, 60], dvdtlim=[-80, 320],
                        color=sns.color_palette("muted").as_hex()[1],
                        save_filepath=None, rasterized=False):
    t0 = features['spikes_threshold_t'][0]
    ap_window = [t0 + x * 0.001 for x in window]

    if len(features['spikes_sweep_id']) > 1 and \
        features['spikes_sweep_id'][1] == features['spikes_sweep_id'][0]:
            ap_window[1] = min(ap_window[1], features['spikes_threshold_t'][1])

    start, end = [ft.find_time_index(data['t'], x) for x in ap_window]
    t = (data['t'][start:end] - data['t'][start]) * 1000 + window[0]
    v = data['voltage'][features['rheobase_index']][start:end]
    # dvdt = ft.calculate_dvdt(v, t, filter=filter) * 1000
    dvdt = ft.calculate_dvdt(v, data['t'][start:end], filter)  # filter=10 or 5

    mpl.rcParams.update(mpl.rcParamsDefault)
    fig, ax = plt.subplots(1,1,figsize=figsize)
    ax.plot(v[0:-1], dvdt, color=color, lw=2 * lw_scale, rasterized=rasterized)

    ax.set_xlim(vlim)
    ax.set_ylim(dvdtlim)
    fig.gca().spines['right'].set_visible(False)
    fig.gca().spines['top'].set_visible(False)
    ax.set_xlabel('Voltage (mV)', fontsize=18)
    ax.set_ylabel('dV/dt (V/s)', fontsize=18)
    ax.yaxis.set_label_coords(-0.22 * 4 / figsize[0],0.5)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 14)
    ax.patch.set_alpha(0)
    ax.grid(False)
    for loc in ['top', 'right', 'bottom', 'left']:
        ax.spines[loc].set_visible(False)
    fig.tight_layout()

    if save_filepath is not None:
        fig.savefig(save_filepath, dpi=200)
    return fig


def plot_first_spike_dvdt(data, features, time_zero='threshold', figsize=(4,4),
                    filter_dvdt=10,
                    window=None, ylim=None,
                    color="gray",
                    save_filepath = None, rasterized=False):
    '''
    Plot dv/dt of the first action potential. Time window is something like:
    Inputs
    -----
    data: raw data of sweeps loaded by load_current_step()
    features: dictionary from extract_istep_features()
    time_zero: whether to use threshold or peak time
    window: time range in ms. such as [t-10, t+40] ms

    Returns
    -------
    figure object
    '''
    assert(time_zero in ['threshold', 'peak'])
    if time_zero == 'threshold':
        t0 = features['spikes_threshold_t'][0]
        if window is None:
            window = [-10, 40]
    elif time_zero == 'peak':
        t0 = features['spikes_peak_t'][0]
        if window is None:
            window = [-15, 35]

    ap_window = [t0 + x * 0.001 for x in window]
    start, end = [ft.find_time_index(data['t'], x) for x in ap_window]
    t = (data['t'][start:end] - data['t'][start]) * 1000 + window[0]
    v = data['voltage'][features['rheobase_index']][start:end]

    #dvdt = ft.calculate_dvdt(v, t, filter=filter_dvdt) * 1000  need filter=0.01 or 0.005
    dvdt = ft.calculate_dvdt(v, data['t'][start:end], filter_dvdt)  # filter=10 or 5

    mpl.rcParams.update(mpl.rcParamsDefault)
    fig, ax = plt.subplots(1,1,figsize=figsize)
    ax.plot(t[:-1], dvdt, color=color, lw=2, rasterized=rasterized)

    threshold_time = (features['spikes_threshold_t'][0] - t0) * 1000
    dvdt_thres_index = ft.find_time_index(data['t'], features['spikes_threshold_t'][0]) - start
    ax.hlines(dvdt[dvdt_thres_index], window[0], threshold_time,
                linestyles='dotted', color='grey')

    ax.set_ylim(ylim)
    fig.gca().spines['right'].set_visible(False)
    fig.gca().spines['top'].set_visible(False)
    ax.set_ylabel('dv/dt (mV/ms)', fontsize=18)
    ax.set_xlabel('Time (ms)', fontsize=18)
    ax.yaxis.set_label_coords(-0.22 * 4 / figsize[0],0.5)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 14)
    ax.patch.set_alpha(0)
    ax.grid(False)
    for loc in ['top', 'right', 'bottom', 'left']:
        ax.spines[loc].set_visible(False)
    fig.tight_layout()

    if save_filepath is not None:
        fig.savefig(save_filepath, dpi=200)
    return fig



def plot_first_spike_2nd_derivative(data, features, time_zero='threshold', figsize=(4,4),
                    filter_dvdt=10,
                    window=None, ylim=None,
                    color="gray",
                    save_filepath = None, rasterized=False):
    '''
    Plot dv/dt of the first action potential. Time window is something like:
    Inputs
    -----
    data: raw data of sweeps loaded by load_current_step()
    features: dictionary from extract_istep_features()
    time_zero: whether to use threshold or peak time
    window: time range in ms. such as [t-10, t+40] ms

    Returns
    -------
    figure object
    '''
    assert(time_zero in ['threshold', 'peak'])
    if time_zero == 'threshold':
        t0 = features['spikes_threshold_t'][0]
        if window is None:
            window = [-10, 40]
    elif time_zero == 'peak':
        t0 = features['spikes_peak_t'][0]
        if window is None:
            window = [-15, 35]

    ap_window = [t0 + x * 0.001 for x in window]
    start, end = [ft.find_time_index(data['t'], x) for x in ap_window]
    t = (data['t'][start:end] - data['t'][start]) * 1000 + window[0]
    v = data['voltage'][features['rheobase_index']][start:end]

    #dvdt = ft.calculate_dvdt(v, t, filter=filter_dvdt) * 1000
    #d2vdt2 = ft.calculate_dvdt(dvdt, t[:-1], filter=filter_dvdt) * 1000
    dvdt = ft.calculate_dvdt(v, data['t'][start:end], filter_dvdt)
    d2vdt2 = ft.calculate_dvdt(dvdt, data['t'][start:end-1], filter_dvdt)

    mpl.rcParams.update(mpl.rcParamsDefault)
    fig, ax = plt.subplots(1,1,figsize=figsize)
    ax.plot(t[:-2], d2vdt2, color=color, lw=1.5, rasterized=rasterized)

    threshold_time = (features['spikes_threshold_t'][0] - t0) * 1000
    dvdt_thres_index = ft.find_time_index(data['t'], features['spikes_threshold_t'][0]) - start
    ax.hlines(d2vdt2[dvdt_thres_index], window[0], threshold_time,
                linestyles='dotted', color='grey')

    ax.set_ylim(ylim)
    fig.gca().spines['right'].set_visible(False)
    fig.gca().spines['top'].set_visible(False)
    ax.set_ylabel('d2v/dt2 (mV/ms^2)', fontsize=18)
    ax.set_xlabel('Time (ms)', fontsize=18)
    ax.yaxis.set_label_coords(-0.22 * 4 / figsize[0],0.5)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 14)
    ax.patch.set_alpha(0)
    ax.grid(False)
    for loc in ['top', 'right', 'bottom', 'left']:
        ax.spines[loc].set_visible(False)
    fig.tight_layout()

    if save_filepath is not None:
        fig.savefig(save_filepath, dpi=200)
    return fig

def combine_vertical(images, scale = 1):
    # combine multiple PIL images
    # roughtly same width
    height = sum([x.size[1] for x in images])
    width = max([x.size[0] for x in images])
    combined = Image.new('RGB', (width, height), (255,255,255))

    y_offset = 0
    for im in images:
        if len(im.split()) > 3:
            combined.paste(im, (0, y_offset), mask=im.split()[3])
        else:
            combined.paste(im, (0, y_offset))
        y_offset += im.size[1]
    if scale != 1:
        combined = combined.resize([int(x * scale) for x in combined.size], resample=Image.BICUBIC)
    return combined


def combine_horizontal(images, scale = 1, same_size = False):
    # combine multiple PIL images
    if not same_size:
        min_height = min([x.size[1] for x in images])
        min_i = np.argmin([x.size[1] for x in images])
        scales = [min_height / x.size[1] for i, x in enumerate(images)]
        resized = images.copy()

        for i in range(len(resized)):
            if i != min_i:
                resized[i] = resized[i].resize([int(x * scales[i]) for x in resized[i].size], resample=Image.BICUBIC)
    else:
        resized = images

    width = sum([x.size[0] for x in resized])
    height = max([x.size[1] for x in resized])
    combined = Image.new('RGB', (width, height), (255,255,255))

    x_offset = 0
    for im in resized:
        if len(im.split()) > 3:
            combined.paste(im, (x_offset,0), mask=im.split()[3])
        else:
            combined.paste(im, (x_offset,0))
        x_offset += im.size[0]
    if scale != 1:
        combined = combined.resize([int(x * scale) for x in combined.size], resample=Image.BICUBIC)

    return combined


def draw_text_on_image(image, text_list, location_list=[(0,0)], font_path='Arial.ttf', font_size=20):
    image = image.copy()
    font = ImageFont.truetype(font_path, size=font_size)
    d = ImageDraw.Draw(image)
    for text, location in zip(text_list, location_list):
        d.text(location, text, font=font, fill=(0,0,0))
    return image

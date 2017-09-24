import os
import numpy as np
import pandas as pd
import six
from collections import OrderedDict

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import gridspec, animation

import stfio

def load_current_step(abf_file, filetype='abf', channels=[0,1]):
    '''
    Load current clamp recordings from pClamp .abf files
    '''
    ch0, ch1 = channels[0], channels[1]
    rec = stfio.read(abf_file)
    assert((rec[ch0].yunits in ['mV', 'pA']) and (rec[ch1].yunits in ['mV', 'pA']))

    data = OrderedDict()
    data['file_id'] = os.path.basename(abf_file).strip('.' + filetype)
    data['file_directory'] = os.path.dirname(abf_file)
    data['record_date'] = rec.datetime.date()
    data['record_time'] = rec.datetime.time()

    data['dt'] = rec.dt / 1000
    data['hz'] = 1./rec.dt * 1000
    data['time_unit'] = 's'

    data['n_channels'] = len(rec)
    data['channel_names'] = [rec[x].name for x in channels]
    data['channel_units'] = [rec[x].yunits for x in channels]
    data['n_sweeps'] = len(rec[ch0])
    data['sweep_length'] = len(rec[ch0][0])

    data['t'] = np.arange(0, data['sweep_length']) * data['dt']

    if rec[ch0].yunits == 'mV' and rec[ch1].yunits == 'pA':
        data['voltage'] = rec[ch0]
        data['current'] = rec[ch1]
    elif rec[ch1].yunits == 'mV' and rec[ch0].yunits == 'pA':
        data['voltage'] = rec[ch1]
        data['current'] = rec[ch0]
    else:
        raise ValueError("channel y-units must be 'mV' or 'pA'.")
    data['voltage'] = [x.asarray() for x in data['voltage']]
    data['current'] = [x.asarray() for x in data['current']]

    return data


def plot_current_step(data, fig_height=6, x_scale=3.5, xlim=[0.3,3.2],
                        startend=None, offset=[0.2, 0.4],
                        blue_sweep=None, vlim=[-155,55], ilim=[-80,10],
                        spikes_sweep_id = None, spikes_threshold_t = None,
                        skip_sweep=1, skip_point=10, save=False):
    '''
    Plot overlayed sweeps in current clamp protocol, with one sweep in blue color
    If detected spikes are provided, also plot detected spikes.
    '''

    plt.style.use('ggplot')

    fig_width = fig_height
    if (spikes_sweep_id is not None) and (spikes_threshold_t is not None):
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
    fig.patch.set_alpha(0.1)
    gs = gridspec.GridSpec(n_plots, 1, height_ratios=height_ratios)

    axes = [plt.subplot(gs[x]) for x in range(n_plots)]

    indices = [x for x in range(data['n_sweeps']) if x % skip_sweep ==0 or x == data['n_sweeps']-1]
    # print(indices)

    if blue_sweep is not None:
        assert(type(blue_sweep) is int)
        if not blue_sweep in indices:
            indices.append(blue_sweep)
    else:
        blue_sweep = indices[-2]

    for i in indices:
        if i == blue_sweep or i == data['n_sweeps'] + blue_sweep:
            color = 'deepskyblue'
            lw=1.25
            size=8
            alpha=1
        else:
            color = 'gray'
            lw=0.2
            size=3
            alpha=0.6

        axes[-2].plot(data['t'][::skip_point], data['voltage'][i][::skip_point], color=color, lw=lw, alpha=alpha)

        axes[-2].set_ylim(vlim)
        axes[-2].set_ylabel('Membrane Voltage (mV)')
        axes[-2].set_xticklabels([])

        axes[-1].plot(data['t'][::skip_point], data['current'][i][::skip_point], color=color, lw=lw, alpha=alpha)
        axes[-1].set_ylim(ilim)
        axes[-1].set_ylabel('Current (pA)')
        axes[-1].set_xlabel('Time (s)')

        if n_plots == 3:
            spikes = spikes_threshold_t[spikes_sweep_id==i]
            axes[0].scatter(spikes, np.ones_like(spikes) * i, marker='o', s=size, c=color, alpha=alpha)
            axes[0].set_ylim([1, data['n_sweeps']])
            axes[0].set_xticklabels([])
            # axes[0].set_ylabel('Sweeps')


    for ax in axes:
        ax.set_xlim(xlim)
        ax.patch.set_alpha(0.2)

    plt.tight_layout()
    if save is True:
        plt.savefig(os.path.join(data['file_directory'], data['file_id']) + '.png', dpi=300)
        plt.savefig(os.path.join(data['file_directory'], data['file_id']) + '.svg')
        plt.savefig(os.path.join(data['file_directory'], data['file_id']) + '.pdf')

    return fig


def animate_current_step(data, fig_height=6, x_scale=3.5, xlim=[0.3,3.2],
                        startend=None, offset=[0.2, 0.4],
                        blue_sweep=None, vlim=[-155,55], ilim=[-80,100],
                        spikes_sweep_id = None, spikes_threshold_t = None,
                        skip_point=10, save=False, save_filepath=None):
    '''
    Make animated GIF containing all the sweeps in current clamp protocol.
    If detected spikes are provided, also plot detected spikes.
    '''
    fig_width = fig_height
    if (spikes_sweep_id is not None) and (spikes_threshold_t is not None):
        fig_height *= 4.0/3.0
        n_plots = 3
        height_ratios = [1,3,1]
    else:
        n_plots = 2
        height_ratios = [3,1]


    def init_animation():
        animate(-1)

    def animate(j):
        plt.style.use('ggplot')

        gs = gridspec.GridSpec(n_plots, 1, height_ratios=height_ratios)
        axes = [plt.subplot(gs[x]) for x in range(n_plots)]

        for i in range(data['n_sweeps']):
            if i == j:
                color = 'deepskyblue'
                lw=1.5
                size=12
                alpha=1
            else:
                color = 'gray'
                lw=0.2
                size=4
                alpha=0.6

            axes[-2].plot(data['t'][::skip_point], data['voltage'][i][::skip_point], color=color, lw=lw, alpha=alpha)

            axes[-2].set_ylim(vlim)
            axes[-2].set_ylabel('Membrane Voltage (mV)')
            axes[-2].set_xticklabels([])
            axes[-1].plot(data['t'][::skip_point], data['current'][i][::skip_point], color=color, lw=lw, alpha=alpha)
            axes[-1].set_ylim(ilim)
            axes[-1].set_ylabel('Current (pA)')
            axes[-1].set_xlabel('Time (s)')
            if n_plots == 3:
                spikes = spikes_threshold_t[spikes_sweep_id==i]
                axes[0].scatter(spikes, np.ones_like(spikes) * i, marker='o', s=size, c=color, alpha=alpha)
                axes[0].set_ylim([1, data['n_sweeps']])
                axes[0].set_xticklabels([])
                # axes[0].set_ylabel('Sweeps')

        for ax in axes:
            ax.set_xlim(xlim)
            ax.patch.set_alpha(0.2)

        plt.tight_layout()

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

    anim = animation.FuncAnimation(fig, animate, init_func=init_animation, frames=data['n_sweeps'])
    if save:
        if save_filepath is not None:
            anim.save(save_filepath, writer='imagemagick', fps=2.5)
        else:
            anim.save(os.path.join(data['file_directory'], data['file_id']) + '.gif', writer='imagemagick', fps=2.5)
    return fig



def plot_fi_curve(stim_amp, firing_rate, save_filepath = None):
    '''
    Plot F-I curve
    '''
    mpl.rcParams.update(mpl.rcParamsDefault)
    fig, ax = plt.subplots(1,1,figsize=(4,3))
    ax.plot(stim_amp, firing_rate, marker='o', linewidth=1.5, markersize=8)
    fig.gca().spines['right'].set_visible(False)
    fig.gca().spines['top'].set_visible(False)
    ax.set_ylabel('Spikes per second')
    ax.set_xlabel('Current injection (pA)')
    fig.tight_layout()
    if save_filepath is not None:
        fig.savefig(save_filepath, dpi=300)
    return fig

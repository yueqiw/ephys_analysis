import os

import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import animation

import stfio


def load_current_step(abf_file, filetype='abf', channels=[0,1]):
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


def plot_current_step(data, figsize=(6,6), xlim=[0.3,3.2], vlim=[-130,50], ilim=[-80,60],
                      skip_sweep=3, skip_point=10, save=False):
    plt.style.use('ggplot')

    fig = plt.figure(figsize=figsize)
    fig.patch.set_alpha(0.1)
    gs = gridspec.GridSpec(2, 1, height_ratios=[3,1])

    axes = [plt.subplot(gs[x]) for x in range(2)]


    t = data['voltage']
    indices = [x for x in range(data['n_sweeps']) if x % skip_sweep ==0 or x == data['n_sweeps']-1]
    print(indices)
    for i in indices:
        if i == indices[-2]:
            color = 'deepskyblue'
            lw=1
            alpha=1
        else:
            color = 'gray'
            lw=0.2
            alpha=0.6

        axes[0].plot(data['t'][::skip_point], data['voltage'][i][::skip_point], color=color, lw=lw, alpha=alpha)

        axes[0].set_ylim(vlim)
        axes[1].plot(data['t'][::skip_point], data['current'][i][::skip_point], color=color, lw=lw, alpha=alpha)
        axes[1].set_ylim(ilim)


    for ax in axes:
        ax.set_xlim(xlim)
        ax.patch.set_alpha(0.2)

    plt.tight_layout()
    if save is True:
        plt.savefig(os.path.join(data['file_directory'], data['file_id']) + '.png')
        plt.savefig(os.path.join(data['file_directory'], data['file_id']) + '.svg')
        plt.savefig(os.path.join(data['file_directory'], data['file_id']) + '.pdf')

    return fig


def animate_current_step(data, figsize=(6,6), xlim=[0.3,3.2], vlim=[-130,50], ilim=[-80,60],
                         skip_sweep=3, skip_point=10, save=False):

    def init_animation():
        animate(-1)

    def animate(j):
        plt.style.use('ggplot')

        gs = gridspec.GridSpec(2, 1, height_ratios=[3,1])
        axes = [plt.subplot(gs[x]) for x in range(2)]

        for i in range(data['n_sweeps']):
            if i == j:
                color = 'deepskyblue'
                lw=1.5
                alpha=1
            else:
                color = 'gray'
                lw=0.2
                alpha=0.6

            axes[0].plot(data['t'][::skip_point], data['voltage'][i][::skip_point], color=color, lw=lw, alpha=alpha)

            axes[0].set_ylim(vlim)
            axes[1].plot(data['t'][::skip_point], data['current'][i][::skip_point], color=color, lw=lw, alpha=alpha)
            axes[1].set_ylim(ilim)


        for ax in axes:
            ax.set_xlim(xlim)
            ax.patch.set_alpha(0.2)

        plt.tight_layout()

    fig = plt.figure(figsize=figsize)

    anim = animation.FuncAnimation(fig, animate, init_func=init_animation, frames=15)
    anim.save(os.path.join(data['file_directory'], data['file_id']) + '.gif', writer='imagemagick', fps=2.5)
    return

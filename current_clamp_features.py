import numpy as np
import logging
import six
from collections import OrderedDict
from allensdk_0_14_2 import ephys_extractor as efex
from allensdk_0_14_2 import ephys_features as ft


def extract_istep_features(data, start, end, subthresh_min_amp = -100, n_subthres_sweeps = 4,
                            sag_target = -100, hero_delta_mV = 10,
                            filter=10., dv_cutoff=5., max_interval=0.02, min_height=5.,
                            min_peak=-30., thresh_frac=0.05, baseline_interval=0.1,
                            baseline_detect_thresh = 0.3):

    '''
    Compute the cellular ephys features from square pulse current injections.

    Note that some default params are different from AllenSDK.
    dv_cutoff 20 -> 6 to catch slower APs in immature neurons.
    max_interval 0.005 -> 0.01 to catch slower APs.
    min_height 2 -> 5 to reduce false positive due to relaxed dv_cutoff
    min_peak -30 -> -25
    '''
    if filter * 1000 >= data['hz']:
        filter = None

    istep_ext = efex.EphysSweepSetFeatureExtractor(
                                [data['t']]*data['n_sweeps'],
                                data['voltage'],
                                data['current'],
                                start=start, end=end,
                                filter=filter, dv_cutoff=dv_cutoff,
                                max_interval=max_interval, min_height=min_height,
                                min_peak=min_peak, thresh_frac=thresh_frac,
                                baseline_interval=baseline_interval,
                                baseline_detect_thresh=baseline_detect_thresh,
                                id_set=list(range(data['n_sweeps'])))

    # only extract long_sqaures features
    fex = efex.EphysCellFeatureExtractor(None, None, istep_ext,
                                            subthresh_min_amp=subthresh_min_amp,
                                            n_subthres_sweeps=n_subthres_sweeps,
                                            sag_target=sag_target)
    fex.process(keys = "long_squares")

    # To make dict-conversion work
    fex._features["short_squares"]["common_amp_sweeps"] = []
    fex._features["ramps"]["spiking_sweeps"] = []
    cell_features = fex.as_dict()

    cell_features = cell_features["long_squares"]

    # find hero sweep for AP train
    # target hero sweep as the first sweep with current amplitude > min threshold_v
    # min threshold is the rheobase + current to increase Vm by 10 mV (10 mV / input_r)
    if cell_features['rheobase_i'] is None:
        has_AP = False
        hero_sweep = None
    else:
        has_AP = True
        rheo_amp = cell_features['rheobase_i']
        input_r = cell_features['input_resistance']

        hero_stim_min = rheo_amp + hero_delta_mV / input_r * 1000
        # print(rheo_amp, hero_delta_mV / input_r * 1000, hero_stim_min)  # DEBUG
        hero_amp = float("inf")
        hero_sweep = None

        for sweep in fex.long_squares_features("spiking").sweeps():
            nspikes = len(sweep.spikes())
            amp = sweep.sweep_feature("stim_amp")
            # print(amp, sweep.sweep_feature("i_baseline"))  # DEBUG

            if nspikes > 0 and amp > hero_stim_min and amp < hero_amp:
                hero_amp = amp
                hero_sweep = sweep

    if hero_sweep:
        adapt = hero_sweep.sweep_feature("adapt")
        latency = hero_sweep.sweep_feature("latency")
        median_isi = hero_sweep.sweep_feature("median_isi")
        mean_rate = hero_sweep.sweep_feature("avg_rate")
    else:
        print("Could not find hero sweep.")

    v_baseline = np.mean(fex.long_squares_features().sweep_features('v_baseline'))
    bias_current = np.mean(fex.long_squares_features().sweep_features('i_baseline'))

    first_spike = cell_features['rheobase_sweep']['spikes'][0] if has_AP else {}
    cell_features['v_baseline'] = v_baseline
    cell_features['bias_current'] = bias_current
    cell_features['hero_sweep'] = hero_sweep.as_dict() if hero_sweep else {}
    cell_features['hero_sweep_stim_amp'] = cell_features['hero_sweep']['stim_amp'] if hero_sweep else None
    cell_features['hero_sweep_index'] = cell_features['hero_sweep']['id'] if hero_sweep else None
    cell_features['first_spike'] = first_spike if has_AP else None

    spikes_sweep_id = []
    spikes_threshold_t = []
    spikes_peak_t = []
    for swp in cell_features['spiking_sweeps']:
        for spike in swp['spikes']:
            spikes_threshold_t.append(spike['threshold_t'])
            spikes_sweep_id.append(swp['id'])
            spikes_peak_t.append(spike['peak_t'])

    spikes_threshold_t = np.array(spikes_threshold_t)
    spikes_sweep_id = np.array(spikes_sweep_id)
    spikes_peak_t = np.array(spikes_peak_t)

    adapt_avg = calculate_adapt(spikes_sweep_id, spikes_peak_t, start, end,
                                    adapt_interval=1.0, max_isi_ratio=2.5)


    summary_features = OrderedDict([
                        ('file_id', data['file_id']),
                        ('has_ap', has_AP),
                        ('v_baseline', v_baseline),
                        ('bias_current', bias_current),
                        ('tau', cell_features['tau'] * 1000),
                        ('capacitance' , cell_features['tau'] / cell_features['input_resistance'] * 10**6 \
                                        if cell_features['input_resistance'] > 0 else None),
                        ('input_resistance', cell_features['input_resistance'] \
                                        if cell_features['input_resistance'] > 0 else None),
                        ('f_i_curve_slope', cell_features['fi_fit_slope']),
                        ('max_firing_rate', max([swp['avg_rate'] for swp in cell_features['sweeps']])),

                        ('sag', cell_features['sag']),
                        ('vm_for_sag', cell_features['vm_for_sag']),

                        ('ap_threshold', first_spike.get('threshold_v')),
                        ('ap_width', first_spike.get('width') * 1000 if not first_spike.get('width') is None else None),
                        ('ap_height', first_spike['peak_v'] - first_spike['trough_v'] if has_AP else None),
                        ('ap_peak', first_spike.get('peak_v')),

                        ('ap_trough', first_spike.get('trough_v')),
                        ('ap_trough_to_threshold', first_spike['threshold_v'] - first_spike['trough_v'] if has_AP else None),
                        ('ap_peak_to_threshold', first_spike['peak_v'] - first_spike['threshold_v'] if has_AP else None),
                        ('ap_upstroke',first_spike.get('upstroke')),
                        ('ap_downstroke', - first_spike.get('downstroke') if has_AP else None),  # make it positive
                        ('ap_updownstroke_ratio', first_spike.get('upstroke_downstroke_ratio')),

                        ('hs_firing_rate' , mean_rate if hero_sweep else None),
                        ('hs_adaptation' , adapt if hero_sweep else None),
                        ('hs_median_isi' , median_isi if hero_sweep else None),
                        ('hs_latency' , latency * 1000 if hero_sweep else None),

                        ('rheobase_index', cell_features['rheobase_extractor_index']),
                        ('rheobase_stim_amp', cell_features['rheobase_i']),
                        ('hero_sweep_stim_amp', cell_features['hero_sweep_stim_amp']),
                        ('hero_sweep_index', cell_features['hero_sweep_index']),

                        ('all_firing_rate', np.array([swp['avg_rate'] for swp in cell_features['sweeps']])),
                        ('all_stim_amp', np.array([swp['stim_amp'] for swp in cell_features['sweeps']])),
                        ('all_adaptation', np.array([swp.get('adapt', np.nan) for swp in cell_features['sweeps']])),
                        ('all_v_baseline', np.array([swp['v_baseline'] for swp in cell_features['sweeps']])),
                        ('all_median_isi', np.array([swp.get('median_isi', np.nan) for swp in cell_features['sweeps']])),
                        ('all_first_isi', np.array([swp.get('first_isi', np.nan) for swp in cell_features['sweeps']])),
                        ('all_latency', np.array([swp.get('latency', np.nan) for swp in cell_features['sweeps']])),
                        ('spikes_sweep_id', spikes_sweep_id),
                        ('spikes_threshold_t', spikes_threshold_t),
                        ('spikes_peak_t', spikes_peak_t),
                        ('adapt_avg', adapt_avg)

    ])

    return cell_features, summary_features


def calculate_adapt(spikes_sweep_id, spikes_peak_t, start, end, adapt_interval=1.0,
                    min_peaks=3, max_isi_ratio=2.5, firing_rate_target=None):
    if len(spikes_sweep_id) == 0:
        return None
    end_adapt = start + adapt_interval
    sweep_id = spikes_sweep_id[spikes_peak_t < end_adapt]
    peaks_all = spikes_peak_t[spikes_peak_t < end_adapt]

    peaks = dict()
    for k, v in zip(sweep_id, peaks_all):
        if peaks.get(k) is not None:
            peaks[k].append(v)
        else:
            peaks[k] = [v]

    # delete sweeps with <3 spikes
    to_pop = []
    for k in peaks:
        if len(peaks[k]) < min_peaks:
            to_pop.append(k)
    for k in to_pop:
        peaks.pop(k)
    if len(peaks) == 0:
        return None

    # calculate isi
    isi = dict()
    for k, v in peaks.items():
        isi[k] = [x - y for x, y in zip(v[1:], v[:-1])]

    # delete long intervals
    for k, v in isi.items():
        for i in range(1, len(v)):
            if v[i] > v[i-1] * max_isi_ratio:
                isi[k] = v[:i]
                break

    # delete sweeps with <2 isi's
    to_pop = []
    for k in isi:
        if len(isi[k]) < 2:
            to_pop.append(k)
    for k in to_pop:
        isi.pop(k)
    if len(isi) == 0:
        return None

    # only take the first 3 sweeps with isi data
    if len(isi) > 3:
        keys = sorted(list(isi.keys()))[:3]
        isi = {k: isi[k] for k in isi if k in keys}

    # calculate adaptation
    adapt = dict()
    for k, v in isi.items():
        adapt[k] = [(x-y)/(x+y) for x, y in zip(v[1:], v[:-1])]

    #print(adapt)
    # take median adaptation from each sweep, then average the 3 sweeps
    adapt_mean = np.mean([np.median(adapt[k]) for k in adapt])

    return adapt_mean

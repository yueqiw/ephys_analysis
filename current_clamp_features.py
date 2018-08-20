import numpy as np
import logging
import six
from collections import OrderedDict
from allensdk_0_14_2 import ephys_extractor as efex
from allensdk_0_14_2 import ephys_features as ft


def extract_istep_features(data, start, end, subthresh_min_amp = -100, n_subthres_sweeps = 4,
                            sag_target = -100, suprathreshold_target_delta_v = 15,
                            suprathreshold_target_delta_i = 15,
                            latency_target_delta_i = 5,
                            filter=10., dv_cutoff=5., max_interval=0.02, min_height=10,
                            min_peak=-20., thresh_frac=0.05, baseline_interval=0.1,
                            baseline_detect_thresh = 0.3, spike_detection_delay=0.001,
                            adapt_avg_n_sweeps=3, adapt_first_n_ratios=2,
                            sag_range_left=-120, sag_range_right=-95):

    '''
    Compute the cellular ephys features from square pulse current injections.

    Note that some default params are different from AllenSDK.
    dv_cutoff 20 -> 6 to catch slower APs in immature neurons.
    max_interval 0.005 -> 0.01 to catch slower APs.
    min_height 2 -> 10 to reduce false positive due to relaxed dv_cutoff
    min_peak -30 -> -20
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
                                spike_detection_delay=spike_detection_delay,
                                id_set=list(range(data['n_sweeps'])))

    # only extract long_sqaures features
    fex = efex.EphysCellFeatureExtractor(None, None, istep_ext,
                                            subthresh_min_amp=subthresh_min_amp,
                                            n_subthres_sweeps=n_subthres_sweeps,
                                            sag_target=sag_target,
                                            sag_range=[sag_range_left, sag_range_right])
    fex.process(keys = "long_squares")

    # To make dict-conversion work
    fex._features["short_squares"]["common_amp_sweeps"] = []
    fex._features["ramps"]["spiking_sweeps"] = []
    cell_features = fex.as_dict()

    cell_features = cell_features["long_squares"]

    # find hero sweep for AP train
    # target hero sweep as the first sweep with current amplitude > min threshold_i
    # min threshold is the rheobase + current

    if cell_features['rheobase_i'] is None:
        has_AP = False
        hero_sweep = None
    else:
        has_AP = True
        rheo_amp = cell_features['rheobase_i']
        input_r = cell_features['input_resistance']

        # hero_stim_target = rheo_amp + suprathreshold_target_delta_v / input_r * 1000
        hero_stim_target = rheo_amp + suprathreshold_target_delta_i - 1
        latency_stim_target = rheo_amp + latency_target_delta_i
        # print(rheo_amp, hero_stim_target)
        # print(rheo_amp, hero_delta_mV / input_r * 1000, hero_stim_target)  # DEBUG
        hero_amp = float("inf")
        hero_sweep = None

        latency_amp = float("inf")
        latency_sweep = None

        all_spiking_sweeps = sorted(fex.long_squares_features("spiking").sweeps(), key=lambda x: x.sweep_feature("stim_amp"))
        for sweep in all_spiking_sweeps:
            nspikes = len(sweep.spikes())
            amp = sweep.sweep_feature("stim_amp")
            # print(amp, sweep.sweep_feature("i_baseline"))  # DEBUG
            # print(amp)
            if nspikes > 0:
                if amp > hero_stim_target and amp < hero_amp:
                    hero_amp = amp
                    hero_sweep = sweep
                    pre_hero_amp = last_amp
                    pre_hero_sweep = last_sweep
                    break
                last_sweep = sweep
                last_amp = amp

        for sweep in all_spiking_sweeps:
            nspikes = len(sweep.spikes())
            amp = sweep.sweep_feature("stim_amp")
            if nspikes > 0:
                if amp > latency_stim_target:
                    latency_amp = amp
                    latency_sweep = sweep
                    pre_latency_amp = last_latency_amp
                    pre_latency_sweep = last_latency_sweep
                    break
                last_latency_sweep = sweep
                last_latency_amp = amp

        # print(hero_amp)

    if has_AP:
        if hero_sweep:
            adapt = hero_sweep.sweep_feature("adapt")
            hs_latency = hero_sweep.sweep_feature("latency")
            pre_hs_latency = pre_hero_sweep.sweep_feature("latency")
            median_isi = hero_sweep.sweep_feature("median_isi")
            hs_rate = hero_sweep.sweep_feature("avg_rate")
            pre_hs_rate = pre_hero_sweep.sweep_feature("avg_rate")

            avg_hs_latency = ((hero_amp - hero_stim_target) * pre_hs_latency + \
                          (hero_stim_target - pre_hero_amp) * hs_latency) / (hero_amp - pre_hero_amp)
            # print(hero_amp, hero_stim_target, pre_hero_amp)
            # print(hs_latency, avg_latency, pre_hs_latency)
            avg_rate = ((hero_amp - hero_stim_target) * pre_hs_rate + \
                          (hero_stim_target - pre_hero_amp) * hs_rate) / (hero_amp - pre_hero_amp)
        else:
            avg_hs_latency = last_sweep.sweep_feature("latency")
            avg_rate = last_sweep.sweep_feature("avg_rate")
            print("Could not find hero sweep.")

        if latency_sweep:
            latency_above = latency_sweep.sweep_feature("latency")
            latency_below = pre_latency_sweep.sweep_feature("latency")

            avg_rheobase_latency = ((latency_amp - latency_stim_target) * latency_below + \
                          (latency_stim_target - pre_latency_amp) * latency_above) / (latency_amp - pre_latency_amp)
            # print(hero_amp, hero_stim_target, pre_hero_amp)

        else:
            avg_rheobase_latency = last_latency_sweep.sweep_feature("latency")

        #print(latency_below, avg_rheobase_latency, latency_above)



    first_spike = cell_features['rheobase_sweep']['spikes'][0] if has_AP else {}
    cell_features['hero_sweep_stim_target'] = hero_stim_target if hero_sweep else None
    cell_features['hero_sweep'] = hero_sweep.as_dict() if hero_sweep else {}
    cell_features['hero_sweep_stim_amp'] = cell_features['hero_sweep']['stim_amp'] if hero_sweep else None
    cell_features['hero_sweep_index'] = cell_features['hero_sweep']['id'] if hero_sweep else None
    cell_features['first_spike'] = first_spike if has_AP else None


    spikes_sweep_id = np.array([swp['id'] for swp in cell_features['spiking_sweeps'] for spike in swp['spikes']])
    all_spikes = [spike for swp in cell_features['spiking_sweeps'] for spike in swp['spikes']]

    spikes_peak_t = np.array([spike['peak_t'] for spike in all_spikes])
    adapt_avg, adapt_all = calculate_adapt(spikes_sweep_id, spikes_peak_t, start,
                                    adapt_interval=1.0, max_isi_ratio=2.5,
                                    min_peaks=4,
                                    avg_n_sweeps=adapt_avg_n_sweeps,
                                    first_n_adapt_ratios=adapt_first_n_ratios)
    # print(adapt_avg)
    # print(adapt_all)

    summary_features = OrderedDict([
                        ('file_id', data['file_id']),
                        ('has_ap', has_AP),
                        ('v_baseline', cell_features['v_baseline']),
                        ('bias_current', cell_features['bias_current']),
                        ('tau', cell_features['tau'] * 1000),
                        ('capacitance' , cell_features['tau'] / cell_features['input_resistance'] * 10**6 \
                                        if cell_features['input_resistance'] > 0 else None),
                        ('input_resistance', cell_features['input_resistance'] \
                                        if cell_features['input_resistance'] > 0 else None),
                        ('f_i_curve_slope', cell_features['fi_fit_slope']),
                        ('max_firing_rate', max([swp['avg_rate'] for swp in cell_features['sweeps']])),

                        ('sag', cell_features['sag']),
                        ('vm_for_sag', cell_features['vm_for_sag']),
                        ('indices_for_sag', cell_features["indices_for_sag"]),
                        ('sag_sweep_indices', cell_features["sag_sweeps"]),

                        ('ap_threshold', first_spike.get('threshold_v')),
                        ('ap_width', first_spike.get('width') * 1000 if not first_spike.get('width') is None else None),
                        ('ap_height', first_spike['peak_v'] - first_spike['trough_v'] if has_AP else None),
                        ('ap_peak', first_spike.get('peak_v')),

                        ('ap_trough', first_spike.get('trough_v')),
                        ('ap_fast_trough', first_spike.get('fast_trough_v')),
                        ('ap_slow_trough', first_spike.get('slow_trough_v')),
                        ('ap_adp', first_spike.get('adp_v')),
                        ('ap_trough_3w', first_spike.get('trough_3w_v')),
                        ('ap_trough_4w', first_spike.get('trough_4w_v')),
                        ('ap_trough_5w', first_spike.get('trough_5w_v')),
                        ('ap_trough_to_threshold', first_spike['threshold_v'] - first_spike['trough_v'] if has_AP else None),
                        ('ap_trough_4w_to_threshold', first_spike['threshold_v'] - first_spike['trough_4w_v'] if has_AP else None),
                        ('ap_trough_5w_to_threshold', first_spike['threshold_v'] - first_spike['trough_5w_v'] if has_AP else None),
                        ('ap_peak_to_threshold', first_spike['peak_v'] - first_spike['threshold_v'] if has_AP else None),
                        ('ap_upstroke',first_spike.get('upstroke')),
                        ('ap_downstroke', - first_spike.get('downstroke') if has_AP else None),  # make it positive
                        ('ap_updownstroke_ratio', first_spike.get('upstroke_downstroke_ratio')),

                        ('hs_firing_rate' , hs_rate if hero_sweep else None),
                        ('avg_firing_rate' , avg_rate if has_AP else None),
                        ('hs_adaptation' , adapt if hero_sweep else None),
                        ('hs_median_isi' , median_isi if hero_sweep else None),
                        ('hs_latency' , hs_latency * 1000 if hero_sweep else None),
                        ('avg_hs_latency' , avg_hs_latency * 1000 if has_AP else None),
                        ('avg_rheobase_latency' , avg_rheobase_latency * 1000 if has_AP else None),

                        ('rheobase_index', cell_features['rheobase_extractor_index']),
                        ('rheobase_stim_amp', cell_features['rheobase_i']),
                        ('hero_sweep_stim_amp', cell_features['hero_sweep_stim_amp']),
                        ('hero_sweep_index', cell_features['hero_sweep_index']),

                        ('all_firing_rate', np.array([swp['avg_rate'] for swp in cell_features['sweeps']])),
                        ('all_stim_amp', np.array([swp['stim_amp'] for swp in cell_features['sweeps']])),
                        ('input_resistance_vm', cell_features['input_resistance_vm']),
                        ('input_resistance_stim_ap', cell_features['input_resistance_stim_ap']),
                        ('all_adaptation', np.array([swp.get('adapt', np.nan) for swp in cell_features['sweeps']])),
                        ('all_v_baseline', np.array([swp['v_baseline'] for swp in cell_features['sweeps']])),
                        ('all_median_isi', np.array([swp.get('median_isi', np.nan) for swp in cell_features['sweeps']])),
                        ('all_first_isi', np.array([swp.get('first_isi', np.nan) for swp in cell_features['sweeps']])),
                        ('all_latency', np.array([swp.get('latency', np.nan) for swp in cell_features['sweeps']])),
                        ('spikes_sweep_id', spikes_sweep_id),
                        ('spikes_threshold_t', np.array([spike['threshold_t'] for spike in all_spikes])),
                        ('spikes_peak_t', spikes_peak_t),
                        ('spikes_trough_t', np.array([spike['trough_t'] for spike in all_spikes])),
                        ('spikes_threshold_v', np.array([spike['threshold_v'] for spike in all_spikes])),
                        ('spikes_peak_v', np.array([spike['peak_v'] for spike in all_spikes])),
                        ('spikes_trough_v', np.array([spike['trough_v'] for spike in all_spikes])),

                        ('spikes_fast_trough_t', np.array([spike['fast_trough_t'] for spike in all_spikes])),
                        ('spikes_fast_trough_v', np.array([spike['fast_trough_v'] for spike in all_spikes])),
                        ('spikes_slow_trough_t', np.array([spike['slow_trough_t'] for spike in all_spikes])),
                        ('spikes_slow_trough_v', np.array([spike['slow_trough_v'] for spike in all_spikes])),
                        ('spikes_adp_t', np.array([spike['adp_t'] for spike in all_spikes])),
                        ('spikes_adp_v', np.array([spike['adp_v'] for spike in all_spikes])),
                        ('spikes_trough_3w_t', np.array([spike['trough_3w_t'] for spike in all_spikes])),
                        ('spikes_trough_3w_v', np.array([spike['trough_3w_v'] for spike in all_spikes])),
                        ('spikes_trough_4w_t', np.array([spike['trough_4w_t'] for spike in all_spikes])),
                        ('spikes_trough_4w_v', np.array([spike['trough_4w_v'] for spike in all_spikes])),
                        ('spikes_trough_5w_t', np.array([spike['trough_5w_t'] for spike in all_spikes])),
                        ('spikes_trough_5w_v', np.array([spike['trough_5w_v'] for spike in all_spikes])),
                        ('adapt_avg', adapt_avg)

    ])

    return cell_features, summary_features


def calculate_adapt(spikes_sweep_id, spikes_peak_t, start, end=None, adapt_interval=1.0,
                    min_peaks=4, max_isi_ratio=2.5, avg_n_sweeps=3, first_n_adapt_ratios=None,
                    firing_rate_target=None):
    '''
    parameters
    ----------
    max_isi_ratio: filter out long gaps.
    min_peaks: only use sweeps with at least 4 peaks (3 isi's, or 2 adapt ratios).
    avg_n_sweeps: use the first n sweeps that satisfy all constraints to calculate average adaptation ratio.
    first_n_adapt_ratios: for each sweep, only take the first n adaptation ratios for averaging.
                Setting this to None then uses all adaptation ratios from the sweep.
                setting this to 1 or 2 adjust for the fact that many neurons only fire a few (3-4) spikes,
                    allowing comparison between high firing rate neurons with low firing rate neurons.
    '''
    if len(spikes_sweep_id) == 0:
        return None, None
    end_adapt = start + adapt_interval
    sweep_id = spikes_sweep_id[spikes_peak_t < end_adapt]
    peaks_all = spikes_peak_t[spikes_peak_t < end_adapt]

    peaks = dict()
    for k, v in zip(sweep_id, peaks_all):
        if peaks.get(k) is not None:
            peaks[k].append(v)
        else:
            peaks[k] = [v]

    # delete sweeps with < 4 spikes
    to_pop = []
    for k in peaks:
        if len(peaks[k]) < min_peaks:
            to_pop.append(k)
    for k in to_pop:
        peaks.pop(k)
    if len(peaks) == 0:
        return None, None

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
            elif v[i-1] > v[i] * max_isi_ratio:
                isi[k] = v[:i-1]
                break

    # delete sweeps with <3 isi's
    to_pop = []
    for k in isi:
        if len(isi[k]) < min_peaks - 1:
            to_pop.append(k)
    for k in to_pop:
        isi.pop(k)
    if len(isi) == 0:
        return None, None

    # only take the first 3 sweeps with isi data
    # assuming sweep id increments with higher current injection
    if len(isi) > avg_n_sweeps:
        keys = sorted(list(isi.keys()))[:avg_n_sweeps]
        isi = {k: isi[k] for k in isi if k in keys}

    # calculate adaptation
    adapt = dict()
    for k, v in isi.items():
        adapt[k] = [(x-y)/(x+y) for x, y in zip(v[1:], v[:-1])]

    #print(adapt)
    # take median adaptation from each sweep, then average the 3 sweeps
    adapt_all = [np.mean(adapt[k][:first_n_adapt_ratios]) for k in adapt]
    adapt_mean = np.mean(adapt_all)

    return adapt_mean, adapt_all

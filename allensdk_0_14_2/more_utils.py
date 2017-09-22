import numpy as np
import logging
import six
from . import ephys_extractor as efex
from . import ephys_features as ft


def extract_istep_features(data, start, end, subthresh_min_amp = -100, hero_delta_mV = 10):
    istep_ext = efex.EphysSweepSetFeatureExtractor(
                                [data['t']]*data['n_sweeps'],
                                data['voltage'],
                                data['current'],
                                start=start, end=end,
                                dv_cutoff=20., thresh_frac=0.05,
                                id_set=list(range(data['n_sweeps'])))

    # only extract long_sqaures features
    fex = efex.EphysCellFeatureExtractor(None, None, istep_ext,
                                            subthresh_min_amp=subthresh_min_amp)
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
        hero_amp = float("inf")
        hero_sweep = None

        for sweep in fex.long_squares_features("spiking").sweeps():
            nspikes = len(sweep.spikes())
            amp = sweep.sweep_feature("stim_amp")

            if nspikes > 0 and amp > hero_stim_min and amp < hero_amp:
                hero_amp = amp
                hero_sweep = sweep

    if hero_sweep:
        adapt = hero_sweep.sweep_feature("adapt")
        latency = hero_sweep.sweep_feature("latency")
        mean_isi = hero_sweep.sweep_feature("mean_isi")
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


    summary_features = {
                        'file_id': data['file_id'],
                        'has_ap': has_AP,
                        'v_baseline': v_baseline,
                        'bias_current': bias_current,

                        'adaptation' : adapt if hero_sweep else None,
                        'latency' : latency if hero_sweep else None,
                        'avg_isi' : mean_isi if hero_sweep else None,
                        'capacitance' : cell_features['tau'] / cell_features['input_resistance'] * 10**6 \
                                        if cell_features['input_resistance'] else None,
                        'ap_peak': first_spike.get('peak_v'),
                        'ap_threshold': first_spike.get('threshold_v'),
                        'ap_trough': first_spike.get('trough_v'),
                        'ap_upstroke':first_spike.get('upstroke'),
                        'ap_downstroke': first_spike.get('downstroke'),
                        'ap_updownstroke_ratio': first_spike.get('upstroke_downstroke_ratio'),
                        'ap_width': first_spike.get('width'),
                        'ap_height': first_spike['peak_v'] - first_spike['trough_v'] if has_AP else None,
                        'ap_trough_to_threshold': first_spike['threshold_v'] - first_spike['trough_v'] if has_AP else None,
                        'f_i_curve_slope': cell_features['fi_fit_slope'],
                        'input_resistance': cell_features['input_resistance'] if cell_features['input_resistance'] > 0 else None,
                        'rheobase_stim_amp': cell_features['rheobase_i'],
                        'rheobase_index': cell_features['rheobase_extractor_index'],
                        'hero_sweep_stim_amp': cell_features['hero_sweep_stim_amp'],
                        'hero_sweep_index': cell_features['hero_sweep_index'],
                        'sag': cell_features['sag'],
                        'tau': cell_features['tau'],
                        'vm_for_sag': cell_features['vm_for_sag']
    }

    return cell_features, summary_features

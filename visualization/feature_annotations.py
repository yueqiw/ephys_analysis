
feature_name_dict = dict(input_resistance='Input Resistance (MOhm)',
                    log_input_resistance='Input Resistance (MOhm, log)',
                     sag='Sag',
                     capacitance='Membrane Capacitance (pF)',
                     log_capacitance='Membrane Capacitance (pF, log)',
                     v_rest='Resting Vm (mV)',
                    f_i_curve_slope='F-I Curve Slope',
                    ap_threshold='Spike threshold (mV)',
                    ap_width='Spike Width (ms)',
                    log_ap_width='Spike Width (ms, log)',
                    ap_peak_to_threshold='Spike Amplitude (mV)',
                    ap_trough_to_threshold='AHP (slow) Amplitude (mV)',
                    ap_trough_5w_to_threshold='AHP (fast) Amplitude (mV)',
                    ap_upstroke='Spike Upstroke (mV/ms)',
                    ap_updownstroke_ratio='Upstroke-Downstroke Ratio',
                    adapt_avg='Adaptation Index',
                    adapt_imputed_mice='Adaptation Index (imputed)',
                    adapt_imputed_zero='Adaptation Index (zero-filled)',
                    avg_hs_latency='Suprathreshold First Spike Latency (ms)',
                    avg_rheobase_latency='First Spike Latency (ms)',
                    log_avg_rheobase_latency='First Spike Latency (ms, log)',
                    hs_firing_rate='Firing rate (Hz, single)',
                    avg_firing_rate='Firing rate (Hz, average)',
                    log1p_avg_firing_rate='Firing rate (Hz, average, log)',
                    vm_for_sag='Vm for Sag (mV)',
                    cluster='Cluster')


features_noAP = ['input_resistance', 'sag', 'capacitance', 'v_rest']

features_ap = ['f_i_curve_slope', 'ap_threshold',
               'ap_width', 'ap_peak_to_threshold',
               'ap_trough_5w_to_threshold',
               'avg_rheobase_latency']

log_features = ['log_input_resistance', 'log_capacitance', 'log_ap_width', 'log_avg_rheobase_latency']
log1p_features = ['log1p_avg_firing_rate']

log_features_noAP = ['log_input_resistance', 'sag', 'log_capacitance', 'v_rest']
log_features_ap = ['f_i_curve_slope', 'ap_threshold',
               'log_ap_width', 'ap_peak_to_threshold',
               'ap_trough_5w_to_threshold',
               'log_avg_rheobase_latency']

features_noAP_extra = ['tau']
fearures_ap_extra = ['ap_updownstroke_ratio', 'ap_trough', 'ap_peak', 'ap_height', 'hs_latency',
                    'ap_upstroke', 'ap_downstroke', 'avg_hs_latency',
                    'hs_firing_rate', 'max_firing_rate', 'avg_firing_rate',
                    'freq_hs_median_isi', 'freq_hs_first_isi',
                    'ap_trough_to_threshold', 'ap_trough_4w',
                    ]
features_ap_extra_troughs = ['ap_adp',
                             'ap_fast_trough',
                             'ap_slow_trough',
                             'ap_trough_3w',
                             'ap_trough_5w',
                             'ap_trough_4w_to_threshold']

features_adapt = ['adapt_avg']
features_adapt_extra = ['hs_adaptation', 'adapt_imputed_mice','adapt_imputed_zero',
                        'adapt_na', 'hs_median_isi', 'hs_first_isi']
qc_cols = ['v_baseline',
       'bias_current', 'vm_for_sag', 'rheobase_stim_amp', 'hold', 'ra_pre',
       'ra_post', 'start', 'step', 'ra_est', 'cm_est']
id_cols = ['experiment', 'cell', 'recording', 'id', 'date']
meta_cols = ['has_ap', 'strain', 'dob', 'age', 'slicetype']
comment_cols = ['comment', 'response']
stain_cols = ['fill']

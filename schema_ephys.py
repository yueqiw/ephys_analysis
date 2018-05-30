import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image
from collections import OrderedDict

from current_clamp import *
from current_clamp_features import extract_istep_features, feature_name_dict
from read_metadata import *
from file_io import load_current_step
# from pymysql import IntegrityError
import datajoint as dj
schema = dj.schema('yueqi_ephys', locals())

FIG_DIR = 'analysis_current_clamp/figures_plot_recording'

'''
class DjImportedFromDirectory(dj.Imported):
    # Subclass of Imported. Initialize with data directory.
    def __init__(self, directory=''):
        self.directory = directory
        super().__init__()
'''

@schema
class EphysExperimentsForAnalysis(dj.Manual):
    definition = """
    # Ephys experiments (excel files) for analysis
    experiment: varchar(128)    # excel files to use for analysis
    ---
    project: varchar(128)      # which project the data belongs to
    use: enum('Yes', 'No')      # whether to use this experiment
    directory: varchar(256)     # the parent project directory
    """

    def insert_experiment(self, excel_file):
        '''
        Insert new sample ephys metadata from excel to datajoint tables
        '''
        entry_list = pd.read_excel(excel_file)[['experiment', 'project', 'use', 'directory']].dropna(how='any')
        entry_list = entry_list.to_dict('records')
        no_insert = True
        for entry in entry_list:
            if entry['use'] == 'No':
                continue
            self.insert1(row=entry, skip_duplicates=True)
            no_insert = False
            #print("Inserted: " + str(entry))
        if no_insert:
            print("No new entry inserted.")
        return


@schema
class Animals(dj.Imported):
    definition = """
    # Sample metadata
    -> EphysExperimentsForAnalysis
    ---
    id: varchar(128)        # organod ID (use date, but need better naming)
    strain : varchar(128)    # genetic strain
    dob = null: date        # date of birth
    date = null: date       # recording date
    age = null: smallint    # nunmber of days (date - dob)
    slicetype: varchar(128) # what kind of slice prep
    external: varchar(128)  # external solution
    internal: varchar(128)  # internal solution
    animal_comment = '': varchar(256)    # general comments
    """

    def _make_tuples(self, key):
        ephys_exp = (EphysExperimentsForAnalysis() & key).fetch1()
        directory = os.path.expanduser(ephys_exp.pop('directory', None))
        print('Populating for: ', key)
        animal_info, _ = read_ephys_info_from_excel_2017(
                            os.path.join(directory, key['experiment'] + '.xlsx'))
        key['id'] = animal_info['id']
        key['strain'] = animal_info['strain']
        if not pd.isnull(animal_info['DOB']): key['dob'] = animal_info['DOB']
        if not pd.isnull(animal_info['age']): key['age'] = animal_info['age']
        key['date'] = animal_info['date']
        key['slicetype'] = animal_info['type']
        key['external'] = animal_info['external']
        key['internal'] = animal_info['internal']
        if not pd.isnull(animal_info['comment']): key['animal_comment'] = animal_info['comment']
        self.insert1(row=key)
        return


@schema
class PatchCells(dj.Imported):
    definition = """
    # Patch clamp metadata for each cell
    -> EphysExperimentsForAnalysis
    cell: varchar(128)      # cell id
    ---
    rp = null: float        # pipette resistance
    cm_est = null: float    # estimated Cm
    ra_est = null: float    # estimated Ra right after whole-cell mode
    rm_est = null: float    # estimated Rm
    v_rest = null: float     # resting membrane potential
    fluor = '': varchar(128)      # fluorescent label
    fill = 'no': enum('yes', 'no', 'unknown', 'out')  # wether the cell is biocytin filled. Out -- cell came out with pipette.
    cell_external = '': varchar(128)   # external if different from sample metadata
    cell_internal = '': varchar(128)   # internal if different from sample metadata
    depth = '': varchar(128)      # microns beneath slice surface
    location = '': varchar(128)   # spatial location
    """

    def _make_tuples(self, key):
        ephys_exp = (EphysExperimentsForAnalysis() & key).fetch1()
        directory = os.path.expanduser(ephys_exp.pop('directory', None))
        print('Populating for: ', key)
        _, metadata = read_ephys_info_from_excel_2017(
                            os.path.join(directory, key['experiment'] + '.xlsx'))
        if 'params' in metadata.columns:
            old_file = True
            cell_info = parse_cell_info_2017_vertical(metadata)
        else:
            old_file = False
            cell_info = parse_cell_info_2017(metadata)

        for i, row in cell_info.iterrows():
            newkey = {}
            newkey['experiment'] = key['experiment']
            newkey['cell'] = row['cell']
            if not pd.isnull(row['Rp']): newkey['rp'] = row['Rp']
            if not pd.isnull(row['Cm']): newkey['cm_est'] = row['Cm']
            if not pd.isnull(row['Ra']): newkey['ra_est'] = row['Ra']
            if not pd.isnull(row['Vrest']): newkey['v_rest'] = row['Vrest']
            if not pd.isnull(row['depth']): newkey['depth'] = row['depth']

            if not old_file:
                if not pd.isnull(row['fluor']): newkey['fluor'] = row['fluor']
                if not pd.isnull(row['Rm']): newkey['rm_est'] = row['Rm']
                if not pd.isnull(row['external']): newkey['cell_external'] = row['external']
                if not pd.isnull(row['internal']): newkey['cell_internal'] = row['internal']
                if not pd.isnull(row['location']): newkey['location'] = row['location']
                if not pd.isnull(row['fill']):
                    if row['fill'].lower() in ['yes', 'no', 'unknown', 'out']:
                        newkey['fill'] = row['fill'].lower()
                    else:
                        print('"fill" must be yes/no/unknown/out. ')
            #print(newkey)
            self.insert1(row=newkey)
        return

@schema
class EphysRecordings(dj.Imported):
    definition = """
    # Patch clamp metadata for each recording file
    -> EphysExperimentsForAnalysis
    cell: varchar(128)      # cell id
    recording: varchar(128) # recording file name
    ---
    clamp = null : enum('v', 'i')           # voltage or current clamp
    protocol = '' : varchar(128)          # protocols such as gapfree, istep, etc
    hold = null : smallint                  # holding current or voltage
    ra_pre = null : smallint                # estimated Ra before protocol
    ra_post = null : smallint               # estimated Ra after protocol
    compensate = '' : varchar(128)        # percentage of Ra compensation
    gain = null : smallint                  # amplifier gain
    filter = null : smallint                # filter in kHz
    start = null : smallint                 # current step starting current
    step = null : smallint                  # step size of current injection
    stim_strength = '' : varchar(128)     # electrical/optical stimulation strength
    stim_duration = null : smallint         # duration of each stim pulse
    stim_interval = null : smallint         # interval between two consecutive pulses
    response = '' : varchar(256)          # what kind of reponse was observed
    comment = '' : varchar(256)           # general comments
    """

    def _make_tuples(self, key):
        ephys_exp = (EphysExperimentsForAnalysis() & key).fetch1()
        directory = os.path.expanduser(ephys_exp.pop('directory', None))
        print('Populating for: ', key)
        _, metadata = read_ephys_info_from_excel_2017(
                            os.path.join(directory, key['experiment'] + '.xlsx'))
        patch_info = parse_patch_info_2017(metadata)

        for i, row in patch_info.iterrows():
            newkey = {}
            newkey['experiment'] = key['experiment']
            newkey['cell'] = row['cell']
            newkey['recording'] = row['file']

            if not pd.isnull(row['clamp']): newkey['clamp'] = row['clamp'].lower()
            if not pd.isnull(row['protocol']): newkey['protocol'] = row['protocol']
            if not pd.isnull(row['hold']): newkey['hold'] = row['hold']
            if not pd.isnull(row['Ra-pre']):
                if type(row['Ra-pre']) is str:
                    newkey['ra_pre'] = 100
                else:
                    newkey['ra_pre'] = row['Ra-pre']
            if not pd.isnull(row['Ra-post']):
                if type(row['Ra-post']) is str:
                    newkey['ra_post'] = 100
                else:
                    newkey['ra_post'] = row['Ra-post']
            if not pd.isnull(row.get('compensate')): newkey['compensate'] = row['compensate']
            if not pd.isnull(row['gain']): newkey['gain'] = row['gain']
            if not pd.isnull(row['filter']): newkey['filter'] = row['filter']
            if not pd.isnull(row.get('start')): newkey['start'] = row['start']
            if not pd.isnull(row.get('step')): newkey['step'] = row['step']
            if not pd.isnull(row.get('stim strength')): newkey['stim_strength'] = row['stim strength']
            if not pd.isnull(row.get('stim duration')): newkey['stim_duration'] = row['stim duration']
            if not pd.isnull(row.get('stim interval')): newkey['stim_interval'] = row['stim interval']
            if not pd.isnull(row['response']): newkey['response'] = row['response']
            if not pd.isnull(row.get('comment')): newkey['comment'] = row['comment']

            self.insert1(row=newkey)
        return


@schema
class CurrentStepTimeParams(dj.Manual):
    definition = """
    # Time window parameters for current injection (account for different protocol settings)
    -> EphysExperimentsForAnalysis
    ---
    istep_start: float      # current injection starting time (s)
    istep_end_1s: float     # time after 1st second (s) -- use the 1st second for analysis
    istep_end: float        # current injection actual ending time (s)
    istep_duration: float   # current injection duration (s)
    """

    def insert_params(self, excel_file):
        '''
        Insert paramters for current injection
        '''
        experiments = EphysExperimentsForAnalysis().fetch('experiment')
        entry_list = pd.read_excel(excel_file)[['experiment', 'istep_start', 'istep_duration']]
        entry_list = entry_list.dropna(how='any').to_dict('records')
        no_insert = True
        for entry in entry_list:
            if not entry['experiment'] in experiments:
                continue
            entry['istep_end_1s'] = entry['istep_start'] + 1
            entry['istep_end'] = entry['istep_start'] + entry['istep_duration']
            if entry['istep_end'] < entry['istep_end_1s']:
                entry['istep_end_1s'] = entry['istep_end']
            self.insert1(row=entry, skip_duplicates=True)
            no_insert = False
            #print("Inserted: " + str(entry))
        if no_insert:
            print("No new entry inserted.")
        return


@schema
class FeatureExtractionParams(dj.Lookup):
    definition = """
    # Parameters for AllenSDK action potential detection algorithm
    params_id : int        # unique id for parameter set
    ---
    filter = 10 : float              # cutoff frequency for 4-pole low-pass Bessel filter in kHz
    dv_cutoff = 4 : float          # minimum dV/dt to qualify as a spike in V/s (optional, default 20)
    max_interval = 0.02 : float     # maximum acceptable time between start of spike and time of peak in sec (optional, default 0.005)
    min_height = 10 : float         # minimum acceptable height from threshold to peak in mV (optional, default 2)
    min_peak = -20 : float          # minimum acceptable absolute peak level in mV (optional, default -30)
    thresh_frac = 0.05 : float      # fraction of average upstroke for threshold calculation (optional, default 0.05)
    baseline_interval = 0.1 : float     # interval length for baseline voltage calculation (before start if start is defined, default 0.1)
    baseline_detect_thresh = 0.3 : float    # dV/dt threshold for evaluating flatness of baseline region (optional, default 0.3)
    subthresh_min_amp = -80 : float         # minimum subthreshold current, not related to spike detection.
    n_subthres_sweeps = 4 : smallint          # number of hyperpolarizing sweeps for calculating Rin and Tau.
    sag_target = -100 : float           # Use the sweep with peak Vm closest to this number to calculate Sag.
    adapt_avg_n_sweeps = 3 : smallint   # Use the first n sweeps with >=3 isi's to calculate average adaptation ratio.
    adapt_first_n_ratios = 2 : smallint # For each sweep, only average the first n adaptation ratios. If None, average all ratios.
    spike_detection_delay = 0.001 : float   # start detecting spikes at (start + delay) to skip the initial voltage jump.
    """


@schema
class APandIntrinsicProperties(dj.Imported):
    definition = """
    # Action potential and intrinsic properties from current injections
    -> EphysExperimentsForAnalysis
    -> FeatureExtractionParams
    -> CurrentStepTimeParams
    cell: varchar(128)      # cell id
    recording: varchar(128) # recording file name
    ---
    has_ap : enum('Yes', 'No')  # Yes/No
    v_baseline = null : float  # mV
    bias_current = null : float  # pA
    tau = null : float  #
    capacitance = null : float  # pF
    input_resistance = null : float  # MOhm
    f_i_curve_slope = null : float  # no unit
    max_firing_rate = null : float  # Hz

    sag = null : float  # no unit
    vm_for_sag = null : longblob  # mV
    indices_for_sag = null : longblob  # no unit
    sag_sweep_indices = null : longblob  # no unit

    ap_threshold = null : float  # mV
    ap_width = null : float  # half height width (peak to trough), ms
    ap_height = null : float  # peak to trough, mV
    ap_peak = null : float  # mV
    ap_trough = null : float  # mV
    ap_trough_to_threshold = null : float  # AHP amplitude, mV, https://neuroelectro.org/ephys_prop/16/
    ap_peak_to_threshold = null : float  # spike amplitude, mV, https://neuroelectro.org/ephys_prop/5/
    ap_upstroke = null : float  # mV/ms
    ap_downstroke = null : float  # -mV/ms, positive
    ap_updownstroke_ratio = null : float  # no unit

    hs_firing_rate = null : float  # Hz
    hs_adaptation = null : float  # no unit
    hs_median_isi = null : float  # ms
    hs_latency = null : float  # ms

    rheobase_index = null : smallint  # no unit
    rheobase_stim_amp = null : float  # pA
    hero_sweep_index = null : smallint  # no unit
    hero_sweep_stim_amp = null : float  # pA

    all_firing_rate : longblob
    all_stim_amp : longblob
    input_resistance_vm : longblob
    input_resistance_stim_ap : longblob
    all_adaptation : longblob
    all_v_baseline : longblob
    all_median_isi : longblob
    all_first_isi : longblob
    all_latency : longblob

    spikes_sweep_id : longblob
    spikes_threshold_t : longblob
    spikes_peak_t: longblob
    spikes_trough_t: longblob

    adapt_avg = null : float  # average adaptation of the 3 sweeps >= 4Hz (1 sec)

    """

    def _make_tuples(self, key):
        ephys_exp = (EphysExperimentsForAnalysis() & key).fetch1()
        directory = os.path.expanduser(ephys_exp.pop('directory', None))
        # use the first second of current injection for analysis, regardless of the actual duration.
        istep_start, istep_end_1s = \
                (CurrentStepTimeParams() & key).fetch1('istep_start', 'istep_end_1s')

        this_sample = (EphysExperimentsForAnalysis() & key)
        all_istep_recordings = (EphysRecordings()  & "protocol = 'istep'")
        cells, istep_recordings = (all_istep_recordings * this_sample).fetch('cell','recording')

        params = (FeatureExtractionParams() & key).fetch1()
        params_id = params.pop('params_id', None)

        for cell, rec in zip(cells, istep_recordings):
            print('Populating for: ' + key['experiment'] + ' ' + rec)
            abf_file = os.path.join(directory, key['experiment'], rec + '.abf')
            data = load_current_step(abf_file, min_voltage=-140)
            cell_features, summary_features = \
                        extract_istep_features(data, start=istep_start, end=istep_end_1s,
                        **params)
            newkey = summary_features.copy()


            newkey['has_ap'] = 'Yes' if summary_features['has_ap'] else 'No'
            newkey['experiment'] = key['experiment']
            newkey['cell'] = cell
            newkey['recording'] = rec
            newkey['params_id'] = params_id
            # _ = newkey.pop('file_id', None)

            self.insert1(row=newkey, ignore_extra_fields=True)
        return


@schema
class CurrentStepPlots(dj.Imported):
    definition = """
    # Plot current clamp raw sweeps + detected spikes. Save figures locally. Store file path.
    -> APandIntrinsicProperties  # TODO actually does not need to depend on this.
    ---
    istep_simple_pdf_path : varchar(256)
    istep_simple_png_large_path : varchar(256)
    istep_simple_png_mid_path : varchar(256)
    istep_pdf_path : varchar(256)
    istep_png_large_path : varchar(256)
    istep_png_mid_path : varchar(256)
    istep_gif_path : varchar(256)
    istep_mp4_path : varchar(256)
    """

    def _make_tuples(self, key):
        ephys_exp = (EphysExperimentsForAnalysis() & key).fetch1()
        directory = os.path.expanduser(ephys_exp.pop('directory', None))
        rec = key['recording']
        print('Populating for: ' + key['experiment'] + ' ' + rec)
        abf_file = os.path.join(directory, key['experiment'], rec + '.abf')
        data = load_current_step(abf_file, min_voltage=-140)

        istep_start, istep_end = \
                (CurrentStepTimeParams() & key).fetch1('istep_start', 'istep_end')

        params = (FeatureExtractionParams() & key).fetch1()
        params_id = params.pop('params_id', None)

        # figures/istep_plots_params-1/2018-03-30_EP2-15/
        parent_directory = os.path.join(FIG_DIR, 'istep_plots_params-' + str(params_id), key['experiment'])
        if not os.path.exists(os.path.join(directory, parent_directory)):
            os.makedirs(os.path.join(directory, parent_directory))

        # The fetched features only contain AP time points for the 1st second
        features_1s = (APandIntrinsicProperties() & key).fetch1()
        # To get all spike times, recalculate APs using the entire current step
        _ , features = \
                    extract_istep_features(data, start=istep_start, end=istep_end,
                    **params)

        for filetype in ['istep_simple', 'istep', 'istep_animation']:
            target_folder = os.path.join(directory, parent_directory, filetype)
            if not os.path.exists(target_folder):
                os.mkdir(target_folder)

        fig = plot_current_step(data, fig_height=6, startend=[istep_start, istep_end],
                                offset=[0.2, 0.4], skip_sweep=1,
                                blue_sweep=features_1s['hero_sweep_index'],
                                spikes_t = features['spikes_peak_t'],
                                spikes_sweep_id = features['spikes_sweep_id'],
                                bias_current = features['bias_current'],
                                other_features = features,
                                save=False)

        target_folder = os.path.join(parent_directory, 'istep_simple')
        key['istep_simple_pdf_path'] = os.path.join(target_folder, 'istep_simple_' + rec + '.pdf')
        fig.savefig(os.path.join(directory, key['istep_simple_pdf_path']))
        key['istep_simple_png_large_path'] = os.path.join(target_folder, 'istep_simple_large_' + rec + '.png')
        fig.savefig(os.path.join(directory, key['istep_simple_png_large_path']), dpi=300)
        key['istep_simple_png_mid_path'] = os.path.join(target_folder, 'istep_simple_mid_' + rec + '.png')
        fig.savefig(os.path.join(directory, key['istep_simple_png_mid_path']), dpi=150)
        plt.close(fig)

        fig = plot_current_step(data, fig_height=6, startend=[istep_start, istep_end],
                                offset=[0.2, 0.4], skip_sweep=1,
                                blue_sweep=features_1s['hero_sweep_index'],
                                spikes_t = features['spikes_peak_t'],
                                spikes_sweep_id = features['spikes_sweep_id'],
                                bias_current = features['bias_current'],
                                other_features = features,
                                rheobase_sweep = features_1s['rheobase_index'],
                                sag_sweeps = features_1s['sag_sweep_indices'],
                                save=False)

        target_folder = os.path.join(parent_directory, 'istep')
        key['istep_pdf_path'] = os.path.join(target_folder, 'istep_' + rec + '.pdf')
        fig.savefig(os.path.join(directory, key['istep_pdf_path']))
        key['istep_png_large_path'] = os.path.join(target_folder, 'istep_large_' + rec + '.png')
        fig.savefig(os.path.join(directory, key['istep_png_large_path']), dpi=300)
        key['istep_png_mid_path'] = os.path.join(target_folder, 'istep_mid_' + rec + '.png')
        fig.savefig(os.path.join(directory, key['istep_png_mid_path']), dpi=150)
        plt.show()
        plt.close(fig)

        key['istep_gif_path'] = os.path.join(parent_directory, 'istep_animation', 'istep_' + rec + '.gif')
        key['istep_mp4_path'] = os.path.join(parent_directory, 'istep_animation', 'istep_' + rec + '.mp4')
        fig_anim, anim = animate_current_step(data, fig_height=6, startend=[istep_start, istep_end], offset=[0.2, 0.4],
                            spikes_t = features['spikes_peak_t'],
                            spikes_sweep_id = features['spikes_sweep_id'],
                            bias_current = features['bias_current'],
                            save=False, blit = True)
        anim.save(os.path.join(directory, key['istep_gif_path']), writer='imagemagick', fps=2.5, dpi=100)
        anim.save(os.path.join(directory, key['istep_mp4_path']), writer='ffmpeg', fps=2.5, dpi=100)
        plt.close(fig_anim)

        self.insert1(row=key)
        return


@schema
class FICurvePlots(dj.Imported):
    definition = """
    # Plot F-I curve from current clamp recordings. Save figures locally. Store file path.
    -> APandIntrinsicProperties
    ---
    fi_svg_path = '' : varchar(256)
    fi_png_path = '' : varchar(256)
    fi_pdf_path = '' : varchar(256)
    """
    def _make_tuples(self, key):
        ephys_exp = (EphysExperimentsForAnalysis() & key).fetch1()
        directory = os.path.expanduser(ephys_exp.pop('directory', None))
        features = (APandIntrinsicProperties() & key).fetch1()
        if features['has_ap'] == 'No':
            self.insert1(row=key)
            return

        rec = key['recording']
        print('Populating for: ' + key['experiment'] + ' ' + rec)
        params = (FeatureExtractionParams() & key).fetch1()
        params_id = params.pop('params_id', None)
        parent_directory = os.path.join(FIG_DIR, 'istep_plots_params-' + str(params_id), key['experiment'])
        if not os.path.exists(os.path.join(directory, parent_directory)):
            os.makedirs(os.path.join(directory, parent_directory))

        target_folder = os.path.join(directory, parent_directory, 'fi_curve')
        if not os.path.exists(target_folder):
            os.mkdir(target_folder)
        # The fetched features only contain AP time points for the 1st second
        # Only use the 1st second for consistency

        fi_curve = plot_fi_curve(features['all_stim_amp'], features['all_firing_rate'])
        key['fi_png_path'] = os.path.join(parent_directory, 'fi_curve', 'fi_' + rec + '.png')
        key['fi_svg_path'] = os.path.join(parent_directory, 'fi_curve', 'fi_' + rec + '.svg')
        key['fi_pdf_path'] = os.path.join(parent_directory, 'fi_curve', 'fi_' + rec + '.pdf')
        fi_curve.savefig(os.path.join(directory, key['fi_png_path']), dpi=200)
        fi_curve.savefig(os.path.join(directory, key['fi_svg_path']))
        fi_curve.savefig(os.path.join(directory, key['fi_pdf_path']))
        plt.show()
        self.insert1(row=key)
        return


@schema
class VICurvePlots(dj.Imported):
    definition = """
    # Plot V-I curve (hyperpolarizing) from current clamp recordings. Save figures locally. Store file path.
    -> APandIntrinsicProperties
    ---
    vi_svg_path = '' : varchar(256)
    vi_png_path = '' : varchar(256)
    vi_pdf_path = '' : varchar(256)
    """
    def _make_tuples(self, key):
        ephys_exp = (EphysExperimentsForAnalysis() & key).fetch1()
        directory = os.path.expanduser(ephys_exp.pop('directory', None))
        features = (APandIntrinsicProperties() & key).fetch1()
        if features['has_ap'] == 'No':
            self.insert1(row=key)
            return

        rec = key['recording']
        print('Populating for: ' + key['experiment'] + ' ' + rec)
        params = (FeatureExtractionParams() & key).fetch1()
        params_id = params.pop('params_id', None)
        parent_directory = os.path.join(FIG_DIR, 'istep_plots_params-' + str(params_id), key['experiment'])
        if not os.path.exists(os.path.join(directory, parent_directory)):
            os.makedirs(os.path.join(directory, parent_directory))

        target_folder = os.path.join(directory, parent_directory, 'vi_curve')
        if not os.path.exists(target_folder):
            os.mkdir(target_folder)
        # The fetched features only contain AP time points for the 1st second
        # Only use the 1st second for consistency

        vi_curve = plot_vi_curve(features['input_resistance_stim_ap'], features['input_resistance_vm'])
        key['vi_png_path'] = os.path.join(parent_directory, 'vi_curve', 'vi_' + rec + '.png')
        key['vi_svg_path'] = os.path.join(parent_directory, 'vi_curve', 'vi_' + rec + '.svg')
        key['vi_pdf_path'] = os.path.join(parent_directory, 'vi_curve', 'vi_' + rec + '.pdf')
        vi_curve.savefig(os.path.join(directory, key['vi_png_path']), dpi=200)
        vi_curve.savefig(os.path.join(directory, key['vi_svg_path']))
        vi_curve.savefig(os.path.join(directory, key['vi_pdf_path']))
        plt.show()
        self.insert1(row=key)
        return


@schema
class FirstSpikePlots(dj.Imported):
    definition = """
    # Plot first spikes from current clamp recordings. Save figures locally. Store file path.
    -> APandIntrinsicProperties
    ---
    spike_svg_path = '' : varchar(256)
    spike_png_path = '' : varchar(256)
    spike_pdf_path = '' : varchar(256)
    """
    def _make_tuples(self, key):
        ephys_exp = (EphysExperimentsForAnalysis() & key).fetch1()
        directory = os.path.expanduser(ephys_exp.pop('directory', None))
        features = (APandIntrinsicProperties() & key).fetch1()
        if features['has_ap'] == 'No':
            self.insert1(row=key)
            return

        rec = key['recording']
        print('Populating for: ' + key['experiment'] + ' ' + rec)
        params = (FeatureExtractionParams() & key).fetch1()
        params_id = params.pop('params_id', None)
        parent_directory = os.path.join(FIG_DIR, 'istep_plots_params-' + str(params_id), key['experiment'])
        if not os.path.exists(os.path.join(directory, parent_directory)):
            os.makedirs(os.path.join(directory, parent_directory))
        target_folder = os.path.join(directory, parent_directory, 'first_spike')
        if not os.path.exists(target_folder):
            os.mkdir(target_folder)
        # The fetched features only contain AP time points for the 1st second
        # Only use the 1st second for consistency
        abf_file = os.path.join(directory, key['experiment'], rec + '.abf')
        data = load_current_step(abf_file, min_voltage=-140)

        first_spike = plot_first_spike(data, features, time_zero='threshold')
        key['spike_png_path'] = os.path.join(parent_directory, 'first_spike', 'spike_' + rec + '.png')
        key['spike_svg_path'] = os.path.join(parent_directory, 'first_spike', 'spike_' + rec + '.svg')
        key['spike_pdf_path'] = os.path.join(parent_directory, 'first_spike', 'spike_' + rec + '.pdf')
        first_spike.savefig(os.path.join(directory, key['spike_png_path']), dpi=200)
        first_spike.savefig(os.path.join(directory, key['spike_svg_path']))
        first_spike.savefig(os.path.join(directory, key['spike_pdf_path']))
        plt.show()
        self.insert1(row=key)
        return


@schema
class PhasePlanes(dj.Imported):
    definition = """
    # Plot phase planes of first spikes. Save figures locally. Store file path.
    -> APandIntrinsicProperties
    ---
    phase_svg_path = '' : varchar(256)
    phase_png_path = '' : varchar(256)
    phase_pdf_path = '' : varchar(256)
    """
    def _make_tuples(self, key):
        ephys_exp = (EphysExperimentsForAnalysis() & key).fetch1()
        directory = os.path.expanduser(ephys_exp.pop('directory', None))
        features = (APandIntrinsicProperties() & key).fetch1()
        if features['has_ap'] == 'No':
            self.insert1(row=key)
            return

        rec = key['recording']
        print('Populating for: ' + key['experiment'] + ' ' + rec)
        params = (FeatureExtractionParams() & key).fetch1()
        params_id = params.pop('params_id', None)
        parent_directory = os.path.join(FIG_DIR, 'istep_plots_params-' + str(params_id), key['experiment'])
        if not os.path.exists(os.path.join(directory, parent_directory)):
            os.makedirs(os.path.join(directory, parent_directory))
        target_folder = os.path.join(directory, parent_directory, 'phase_plane')
        if not os.path.exists(target_folder):
            os.mkdir(target_folder)
        # The fetched features only contain AP time points for the 1st second
        # Only use the 1st second for consistency
        abf_file = os.path.join(directory, key['experiment'], rec + '.abf')
        data = load_current_step(abf_file, min_voltage=-140)

        phase_plane = plot_phase_plane(data, features, filter=0.005)
        key['phase_png_path'] = os.path.join(parent_directory, 'phase_plane', 'phase_' + rec + '.png')
        key['phase_svg_path'] = os.path.join(parent_directory, 'phase_plane', 'phase_' + rec + '.svg')
        key['phase_pdf_path'] = os.path.join(parent_directory, 'phase_plane', 'phase_' + rec + '.pdf')
        phase_plane.savefig(os.path.join(directory, key['phase_png_path']), dpi=200)
        phase_plane.savefig(os.path.join(directory, key['phase_svg_path']))
        phase_plane.savefig(os.path.join(directory, key['phase_pdf_path']))
        plt.show()
        self.insert1(row=key)
        return


@schema
class CombinedPlots(dj.Imported):
    definition = """
    # Combine F-I, first spike, phase plane and current step plots together.
    -> CurrentStepPlots
    -> FICurvePlots
    -> FirstSpikePlots
    -> PhasePlanes
    ---
    small_fi_spike_phase = '' : varchar(256)
    small_istep_simple_fi_spike_phase = '' : varchar(256)
    small_istep_fi_spike_phase = '' : varchar(256)
    mid_fi_spike_phase = '' : varchar(256)
    mid_istep_simple_fi_spike_phase = '' : varchar(256)
    mid_istep_fi_spike_phase = '' : varchar(256)
    large_fi_spike_phase = '' : varchar(256)
    large_istep_simple_fi_spike_phase = '' : varchar(256)
    large_istep_fi_spike_phase = '' : varchar(256)
    """

    def _make_tuples(self, key):
        ephys_exp = (EphysExperimentsForAnalysis() & key).fetch1()
        directory = os.path.expanduser(ephys_exp.pop('directory', None))
        fi = (FICurvePlots() & key).fetch1('fi_png_path')
        spike = (FirstSpikePlots() & key).fetch1('spike_png_path')
        phase = (PhasePlanes() & key).fetch1('phase_png_path')
        istep_simple = (CurrentStepPlots() & key).fetch1('istep_simple_png_large_path')
        istep = (CurrentStepPlots() & key).fetch1('istep_png_large_path')
        if not (fi and spike and phase and istep and istep_simple):
            self.insert1(row=key)
            return

        rec = key['recording']
        print('Populating for: ' + key['experiment'] + ' ' + rec)
        params = (FeatureExtractionParams() & key).fetch1()
        params_id = params.pop('params_id', None)
        parent_directory = os.path.join(FIG_DIR, 'istep_plots_params-' + str(params_id), key['experiment'])

        left_large = combine_vertical([Image.open(os.path.join(directory, x)) for x in [fi, spike, phase]], scale=1)
        left_mid = left_large.resize([int(x * 0.5) for x in left_large.size], resample=Image.BICUBIC)
        left_small = left_large.resize([int(x * 0.2) for x in left_large.size], resample=Image.BICUBIC)
        simple_large = combine_horizontal([left_large, Image.open(os.path.join(directory, istep_simple))], scale=1)
        simple_mid = simple_large.resize([int(x * 0.5) for x in simple_large.size], resample=Image.BICUBIC)
        simple_small = simple_large.resize([int(x * 0.2) for x in simple_large.size], resample=Image.BICUBIC)
        all_large = combine_horizontal([left_large, Image.open(os.path.join(directory, istep))], scale=1)
        all_mid = all_large.resize([int(x * 0.5) for x in all_large.size], resample=Image.BICUBIC)
        all_small = all_large.resize([int(x * 0.2) for x in all_large.size], resample=Image.BICUBIC)

        for fpath, folder, img in zip(['large_fi_spike_phase', 'mid_fi_spike_phase', 'small_fi_spike_phase',
                            'large_istep_simple_fi_spike_phase', 'mid_istep_simple_fi_spike_phase', 'small_istep_simple_fi_spike_phase',
                            'large_istep_fi_spike_phase', 'mid_istep_fi_spike_phase', 'small_istep_fi_spike_phase'],
                            ['combine_fi_spike_phase'] * 3 + ['combine_istep_simple_fi_spike_phase'] * 3 + ['combine_istep_fi_spike_phase'] * 3,
                            [left_large, left_mid, left_small, simple_large, simple_mid, simple_small, all_large, all_mid, all_small]):
            target_folder = os.path.join(directory, parent_directory, folder)
            if not os.path.exists(target_folder):
                os.mkdir(target_folder)
            key[fpath] = os.path.join(parent_directory, folder, fpath + '_' + rec + '.png')

            img.save(os.path.join(directory, key[fpath]))
        self.insert1(row=key)
        return



@schema
class CombinedPlotsWithText(dj.Imported):
    definition = """
    # Combine F-I, first spike, phase plane and current step plots together.
    -> CurrentStepPlots
    -> FICurvePlots
    -> VICurvePlots
    -> FirstSpikePlots
    -> PhasePlanes
    -> Animals
    -> PatchCells
    -> APandIntrinsicProperties
    ---
    small_fi_vi_spike_phase = '' : varchar(256)
    mid_fi_vi_spike_phase = '' : varchar(256)
    large_fi_vi_spike_phase = '' : varchar(256)
    small_istep_fi_vi_spike_phase = '' : varchar(256)
    mid_istep_fi_vi_spike_phase = '' : varchar(256)
    large_istep_fi_vi_spike_phase = '' : varchar(256)
    """

    def _make_tuples(self, key):
        ephys_exp = (EphysExperimentsForAnalysis() & key).fetch1()
        directory = os.path.expanduser(ephys_exp.pop('directory', None))
        fi = (FICurvePlots() & key).fetch1('fi_png_path')
        vi = (VICurvePlots() & key).fetch1('vi_png_path')
        spike = (FirstSpikePlots() & key).fetch1('spike_png_path')
        phase = (PhasePlanes() & key).fetch1('phase_png_path')
        istep = (CurrentStepPlots() & key).fetch1('istep_png_large_path')
        animal = (Animals() & key).fetch1()
        cell = (PatchCells() & key).fetch1()
        features_1s = (APandIntrinsicProperties() & key).fetch1()
        features_and_meta = OrderedDict()
        features_and_meta.update(animal)
        features_and_meta.update(cell)
        features_and_meta.update(features_1s)

        if not (fi and spike and phase and istep):
            self.insert1(row=key)
            return

        rec = key['recording']
        print('Populating for: ' + key['experiment'] + ' ' + rec)
        params = (FeatureExtractionParams() & key).fetch1()
        params_id = params.pop('params_id', None)
        parent_directory = os.path.join(FIG_DIR, 'istep_plots_params-' + str(params_id), key['experiment'])

        top_large = combine_horizontal([Image.open(os.path.join(directory, x)) for x in [vi, fi]], scale=1)
        bot_large = combine_horizontal([Image.open(os.path.join(directory, x)) for x in [phase, spike]], scale=1)

        left_large = combine_vertical([top_large, bot_large], scale=1)
        left_mid = left_large.resize([int(x * 0.5) for x in left_large.size], resample=Image.BICUBIC)
        left_small = left_large.resize([int(x * 0.2) for x in left_large.size], resample=Image.BICUBIC)

        left_with_text = combine_vertical([top_large, bot_large, Image.new('RGB', bot_large.size, (255,255,255))], scale=1)

        # print metadata and features on the plot
        features_keys = ['input_resistance', 'sag', 'capacitance', 'v_rest',
                             'f_i_curve_slope', 'ap_threshold', 'ap_width', 'ap_peak_to_threshold',
                             'ap_trough_to_threshold', 'ap_upstroke', 'ap_updownstroke_ratio', 'adapt_avg']
        metadata_keys = ['date', 'strain', 'cell', 'recording', 'dob', 'age', 'fill']
        features_to_print = [(feature_name_dict[feature], features_and_meta[feature]) for feature in features_keys]
        #print(features_to_print)
        features_to_print = '\n'.join(["{}: {:.3g}".format(x, y) if isinstance(y, float) else "{}: {}".format(x, y) for x, y in features_to_print])
        metadata_to_print = [(metadata, features_and_meta[metadata]) for metadata in metadata_keys]
        metadata_to_print = '\n'.join(["{}: {}".format(x, y) for x, y in metadata_to_print])
        left_with_text = draw_text_on_image(left_with_text, [metadata_to_print, features_to_print],
                        [(100,1700), (900,1700)], font_size=40)

        all_large = combine_horizontal([left_with_text, Image.open(os.path.join(directory, istep))], scale=1)
        all_mid = all_large.resize([int(x * 0.5) for x in all_large.size], resample=Image.BICUBIC)
        all_small = all_large.resize([int(x * 0.2) for x in all_large.size], resample=Image.BICUBIC)

        for fpath, folder, img in zip(['large_fi_vi_spike_phase', 'mid_fi_vi_spike_phase', 'small_fi_vi_spike_phase',
                            'large_istep_fi_vi_spike_phase', 'mid_istep_fi_vi_spike_phase', 'small_istep_fi_vi_spike_phase'],
                            ['combine_fi_vi_spike_phase'] * 3 + ['combine_istep_fi_vi_spike_phase'] * 3,
                            [left_large, left_mid, left_small, all_large, all_mid, all_small]):
            target_folder = os.path.join(directory, parent_directory, folder)
            if not os.path.exists(target_folder):
                os.mkdir(target_folder)
            key[fpath] = os.path.join(parent_directory, folder, fpath + '_' + rec + '.png')

            img.save(os.path.join(directory, key[fpath]))
        self.insert1(row=key)
        return

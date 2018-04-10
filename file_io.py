
import os
import numpy as np
import pandas as pd
from collections import OrderedDict
import gzip

try:
    import cPickle as pickle  # much faster
except:
    import pickle

try:
    import stfio
    from stfio import StfIOException
except ImportError:
    print("Module stfio is not installed. Make sure .abf files are converted to .pkl in Python 2.")


def load_current_step(abf_file, filetype='abf', channels=[0,1], min_voltage=-140):
    '''
    Load current clamp recordings from pClamp .abf files
    filetype: one of ['abf', 'pkl']
            'pkl' is the pickle file converted from abf
    min_voltage: None or a number (e.g. -130), traces with min below this voltage are not loaded.
    '''
    ch0, ch1 = channels[0], channels[1]
    use_pkl = False
    if filetype == 'abf':
        try:
            rec = stfio.read(abf_file)
        except NameError:
            use_pkl = True
    elif filetype == 'pkl' or use_pkl:
        try:
            with gzip.open(abf_file[:-4] + '.pkl', 'rb') as handle: # compressed pkl
                rec = pickle.load(handle) # for files converted to .pkl
        except IOError:
            with open(abf_file[:-4] + '.pkl', 'rb') as handle:  # non-compressed pkl
                rec = pickle.load(handle) # for files converted to .pkl

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


    if not min_voltage is None:
        to_pop = []
        for i, x in enumerate(data['voltage']):
            if np.min(x) < min_voltage:
                to_pop.append(i)
        data['voltage'] = [x for i, x in enumerate(data['voltage']) if not i in to_pop]
        data['current'] = [x for i, x in enumerate(data['current']) if not i in to_pop]
        data['n_sweeps'] -= len(to_pop)

    return data

def save_data_as_pickle(data, pkl_file, compress=True):
    if compress:
        with gzip.open(pkl_file, 'wb') as handle:
            pickle.dump(data, handle, protocol=2)
    else:
        with open(pkl_file, 'wb') as handle:
            pickle.dump(data, handle, protocol=2)

def current_clamp_abf_to_pkl(in_file, out_file):
    print("Saving: " + out_file)
    data = load_current_step(in_file, channels=[0,1], min_voltage=-140)
    save_data_as_pickle(data, out_file)

def batch_current_clamp_abf_to_pkl(input_folder, output_folder=None):
    if not output_folder is None and input_folder != output_folder:
        try:
            os.makedirs(output_folder)
        except OSError:
            pass
    if output_folder is None:
        output_folder = input_folder

    filenames = [x for x in os.listdir(input_folder) if x.endswith(".abf")]
    input_file_paths = [os.path.join(input_folder, x) for x in filenames]
    output_file_paths = [os.path.join(output_folder, x[:-4] + ".pkl") for x in filenames]
    for in_file, out_file in zip(input_file_paths, output_file_paths):
        current_clamp_abf_to_pkl(in_file, out_file)

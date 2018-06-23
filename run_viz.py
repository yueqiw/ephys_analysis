import os, argparse, json
import pandas as pd
from visualization.iclamp_app import iclamp_viz
#TODO: integrate with datajoint, and fetch data directly from database

if __name__ == '__main__':
    with open("config.txt", 'r') as f:
        file_paths = json.load(f)

    parser = argparse.ArgumentParser(description='Viz for Patch Clamp Electrophysiology')
    parser.add_argument('--data',
                    default=os.path.expanduser(file_paths['data']),
                    type=str, help='Directory for raw data')
    parser.add_argument('--analysis',
                    default=os.path.expanduser(file_paths['analysis']),
                    type=str, help='Directory for analysis files')
    args = parser.parse_args()

    # load data
    # use pre-computed features stored on local drive (fetched from Datajoint.)
    data_spike = pd.read_excel(os.path.join(args.analysis, 'istep_unique_filtered_spike.xlsx'), index_col=0)
    data_spike = data_spike.rename(columns={'recording.1': 'recording'})
    plot_paths = pd.read_csv(os.path.join(args.analysis, 'plot_path.csv'), index_col=0)
    app = iclamp_viz(input_data=data_spike, plot_paths=plot_paths, data_root_dir=args.data)
    app.run_server(host='0.0.0.0', port=1235, debug=True)

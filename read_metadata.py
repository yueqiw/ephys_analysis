import os

import numpy as np
import pandas as pd
from collections import OrderedDict


def read_ephys_info_from_excel_2017(excel_file, skiprows_animal=1, skiprows_cell=5):
    # read Ex and In solutions from the first two lines
    excelname = os.path.basename(excel_file)
    excelname = os.path.splitext(excelname)[0]

    animal_info = pd.read_excel(excel_file, header=0, skiprows=skiprows_animal)[:1]
    animal_info = animal_info[[x for x in animal_info.columns if not 'Unnamed' in x]]
    animal_info.columns = animal_info.columns.str.strip()
    animal_info['excelname'] = excelname

    metadata = pd.read_excel(excel_file, skiprows=skiprows_cell).dropna(how='all')
    metadata = metadata[[x for x in metadata.columns if not 'Unnamed' in x]]
    metadata['cell'] = metadata['cell'].fillna(method='ffill')
    if metadata['cell'].dtype == 'float':
        metadata['cell'] = metadata['cell'].astype('int').astype('str')
    metadata.columns = metadata.columns.str.strip()
    metadata['excelname'] = excelname
    return animal_info.loc[0], metadata


def parse_cell_info_2017_vertical(metadata, params='params', value='value'):
    cell_info = metadata.loc[:,'cell':value].dropna(subset=[params])
    cell_info = cell_info.pivot(index='cell', columns = params, values = value).reset_index()
    cell_info['excelname'] = metadata['excelname'][0]
    return cell_info

def parse_cell_info_2017(metadata, left='cell', right='fill'):
    cell_info = metadata.loc[:,left:right].dropna(subset=['cell'])
    cell_info = cell_info.drop_duplicates(subset=['cell'], keep='first')
    cell_info['excelname'] = metadata['excelname'][0]
    return cell_info

def parse_patch_info_2017(metadata):
    cell_ids = metadata.loc[:,['cell','file']].dropna(subset=['file'])
    patch_info = metadata.loc[:,'file':].dropna(subset=['file', 'protocol'])
    patch_info = pd.merge(cell_ids,patch_info, on='file')
    return patch_info

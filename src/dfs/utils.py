import pandas as pd
import re
import json
import numpy as np


def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def read_dfs_report(file_path):
    report = None

    with open(file_path) as f:
        i = 0
        for line in f:
            line = line.strip('\n')
            if i == 0:
                report = pd.DataFrame(columns=line.split(','))
            else:
                cur_val = ''
                vals = []
                for val_ in line.split(','):
                    if '[' not in val_ and ']' not in val_ and cur_val == '':
                        vals.append(float(val_))
                    elif '[' in val_ and ']' in val_:
                        vals.append([float(re.sub('\[|\]', '', val_))])
                    elif '[' in val_:
                        cur_val += val_[1:]
                    elif ']' in val_:
                        cur_val += val_[:-1]
                        cur_val = [float(x) for x in cur_val.split()]
                        vals.append(cur_val)
                        cur_val = ''

                report.loc[i - 1] = vals

            i += 1

    return report


def read_insilico_data(file_path, shuffle=True):
    df = pd.read_csv(file_path)
    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)
    x = df[df.columns[df.columns != 'y']].values
    y = df['y'].values.astype(float).reshape(len(x), 1)

    return x, y


def read_json(file_path):
    with open(file_path) as f:
        return json.load(f)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

import json
import os
import numpy as np
import glob
import pandas as pd


def read_results(report_name):
    str_report = open(report_name, 'r').read().strip()[:-1]
    return json.loads(str_report + ']')


def printf(*args, indent=0, **kwargs):
    print(' ' * (indent * 4), *args, **kwargs)


class Accumulator:
    def __init__(self):
        self.values = []

    def append(self, v):
        self.values.append(v)

    def max(self):
        return max(self.values)

    def min(self):
        return min(self.values)

    def sum(self):
        return sum(self.values)

    def __len__(self):
        return len(self.values)

    def avg(self):
        return self.sum() / len(self)

    def __repr__(self):
        return f'(avg={self.avg}, {len(self)})'


def extract_reports(report_folder, report_names='baselines'):
    perf_reports = {}
    batch_reports = {}

    for vendor_name in os.listdir(report_folder):
        printf(f'Processing vendor: {vendor_name}')

        baselines_results = glob.glob(f'{report_folder}/{vendor_name}/{report_names}*')

        device_count = len(baselines_results)
        printf(f'Found reports for {device_count} GPUs', indent=1)
        vendor_perf_reports = {}
        vendor_batch_loss_reports = {}

        perf_reports[vendor_name] = vendor_perf_reports
        batch_reports[vendor_name] = vendor_batch_loss_reports

        # we want device 0 to be first since it is the report with the most reports (distributed)
        baselines_results.sort()

        for idx, device_reports in enumerate(baselines_results):
            printf(f'Reading (device_id: {idx}) (report: {device_reports})', indent=2)

            # List of Results for each Trial
            reports = read_results(device_reports)

            for bench_result in reports:
                bench_name = bench_result['name']
                uid = bench_result['unique_id']
                version = bench_result['version']
                unique_id = bench_name  # (uid, bench_name)

                # Select the task that matters
                if bench_name == 'wlm' and bench_result['model'] != 'GRU':
                    continue

                if bench_name == 'wlmfp16' and bench_result['model'] != 'GRU':
                    continue

                if bench_name == 'loader' and bench_result['batch_size'] != 256:
                    continue

                if bench_name == 'toy_lstm' and bench_result['dtype'] != 'float32':
                    continue

                if bench_name == 'ssd' and len(bench_result['vcd']) > 1:
                    continue

                if bench_name == 'image_loading_loader_pytorch_loaders.py':
                    batch_size = bench_result['batch_size']
                    unique_id = f'{unique_id}_{batch_size}'
                    version = f'{unique_id}_{batch_size}'

                printf(f'Processing {bench_name} {version}', indent=3)

                if unique_id in vendor_perf_reports and idx == 0:
                    printf(f'[!] Error two benchmark with the same name (name: {bench_name})', indent=4)
                elif idx == 0:
                    perf_report = dict(
                        train_item=Accumulator(),
                        unique_id=uid,
                        version=version,
                        error=[],
                        name=bench_name,
                    )
                    vendor_perf_reports[unique_id] = perf_report

                elif unique_id not in vendor_perf_reports:
                    printf(f'[!] Error missing benchmark for previous GPU (name: {bench_name})', indent=4)
                    perf_report = dict(train_item=Accumulator(), unique_id=uid, version=version, error=[])
                    vendor_perf_reports[unique_id] = perf_report

                # Accumulate values
                perf_report = vendor_perf_reports[unique_id]
                if perf_report['unique_id'] != uid:
                    printf(f'[!] Error unique_ids do not match cannot aggregate (name: {bench_name})!', indent=4)
                    perf_report['error'].append('id mismatch')

                elif perf_report['version'] != version:
                    printf(f'[!] Error versions do not match cannot aggregate!', indent=4)
                    perf_report['error'].append('version mismatch')

                else:
                    perf_report['train_item'].append(bench_result['train_item']['avg'])

                batch_loss = bench_result.get('batch_loss')
                if batch_loss is None:
                    printf(f'/!\\ No batch loss for benchmark (name: {bench_name})', indent=4)
                else:
                    if unique_id not in vendor_batch_loss_reports:
                        vendor_batch_loss_reports[unique_id] = []

                    vendor_batch_loss_reports[unique_id].append(batch_loss)

    return perf_reports


def filer_report(rep, agg):
    new_rep = {}

    for vendor, report in rep.items():
        vendor_rep = {}
        new_rep[vendor] = vendor_rep

        for name, v in report.items():
            key = f'{name}_{v["version"]}_{v["unique_id"]}'
            vendor_rep[key] = getattr(v['train_item'], agg)()

    return new_rep


weight_table = [
    ('atari'                   , (2.88, 26.5405955792167)),
    ('cart'                    , (2.67, 7302.07868564706)),
    ('convnet_distributed_fp16', (3.16, 787.612513885864)),
    ('convnet_distributed'     , (2.97, 679.552350938073)),
    ('convnet_fp16'            , (2.97, 1679.83933693595)),
    ('convnet'                 , (2.79, 854.372140032470)),
    ('dcgan_all'               , (2.97, 309.723619627068)),
    ('dcgan'                   , (2.79, 953.948799476626)),
    ('fast_style'              , (2.79, 1012.08893408226)),
    ('loader'                  , (2.96, 7399.55789895996)),
    ('recom'                   , (2.81, 74767.2559322286)),
    ('reso'                    , (2.79, 1177.57382438524)),
    ('ssd'                     , (2.79, 145.729436411335)),
    ('toy_lstm'                , (2.67, 4.10197009223690)),
    ('toy_reg'                 , (2.67, 1234013.49127685)),
    ('translator'              , (2.78, 900.443830123957)),
    ('vae'                     , (2.79, 27375.6153865499)),
    ('wlmfp16'                 , (2.96, 22089.7959228754)),
    ('wlm'                     , (2.78, 6487.87603739007)),
]


def load_comparison_data():
    import os
    from milabench import report
    dirname = os.path.dirname(report.__file__)
    return pd.read_csv(f'{dirname}/data.csv', index_col='bench')


def show_perf(folder_name, report_name):
    perf_reports = extract_reports(folder_name, report_name)
    df = pd.DataFrame(filer_report(perf_reports, 'avg'))

    sd = df.std(axis=1)
    df.loc[:, 'result'] = (df.sum(axis=1) - df.max(axis=1) - df.min(axis=1)) / (df.count(axis=1) - 2)
    df.loc[:, 'sd'] = sd
    df.loc[:, 'sd%'] = sd / df.loc[:, 'result']

    return df.loc[:, ('result', 'sd', 'sd%')]


def compute_overall_score(df, col='result'):
    final_report = {}
    total = 0
    wtotal = 0

    for k, value in df[col].items():

        for bk, (w, b) in weight_table:
            if k.startswith(bk):
                v = value * w / b
                final_report[k] = v
                total += v
                wtotal += w
                break

    final_report['total'] = total / wtotal
    return final_report


def check_variance(df):

    scores = []
    for i in ['output2', 'output8', 'output1', 'output10', 'output4', 'output5',
              'output9', 'output7', 'output6', 'output0', 'output3']:

        report = compute_overall_score(df, col=i)
        scores.append(report['total'])

    scores = np.array(scores)
    variance = scores.std() * 100 / scores.mean()

    print(f'Mean: {scores.mean()}')
    print(f'  SD: {scores.std()}')
    print(f' SD%: {variance}')

    # Results must be consistent
    assert variance < 1
    return scores


def other(df):
    report = compute_overall_score(df)
    print(json.dumps(report, indent=2))

    df.to_csv('report.csv')

    check_variance(df)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--reports', type=str, help='folder containing all the reports reports/folder/name.json')
    parser.add_argument('--name', type=str, default='fast', help='name of the report to load')
    parser.add_argument('--show-comparison', action='store_true', default=False)
    parser.add_argument('--gpu-model', choices=['V100', 'MI50', 'RTX'], default=None)
    parser.add_argument('--csv', type=str, default='',
                        help='file to save the results in as CSV')

    args = parser.parse_args()

    df = show_perf(args.reports, args.name)
    print()

    pd.set_option('display.max_colwidth', 80)
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    na_rows =  df[df.isna().any(axis=1)]

    if args.show_comparison:
        baselines = load_comparison_data()

        if args.gpu_model:
            baselines = baselines[args.gpu_model]

        print('=' * 80)
        print(' ' * 12, 'Baselines')
        print('-' * 80)
        print(baselines)
        print('=' * 80, '\n')

    if len(na_rows):
        print('>' * 80)
        print('Dropped Rows because of NaN\n')
        print('/!\\ if there are nans; it means there were not enough observations to compute the standard deviation\n')
        print(na_rows)
        print('<' * 80)

    df = df.dropna()

    if args.gpu_model:
        baselines = load_comparison_data()[args.gpu_model]

        names = []
        for k in df.index:
            name, _, _, = k.rsplit('_', maxsplit=2)
            names.append(name)

        df.insert(0, 'bench', names)
        df = df.set_index('bench')

        df['target'] = baselines
        df['diff'] = (df['result'] - df['target']) / df['target']

    print('--')
    df = df.ix[:, ('target', 'result', 'sd', 'sd%', 'diff')]

    if args.csv:
        with open(args.csv, 'w') as f:
            f.write(df.to_csv())

    print(df)
    print('--')

    # Compute Scores
    # --------------
    perf_score = None
    sd_quantile = df["sd%"].quantile(0.80) * 100
    sd_sd = df["sd%"].std() * 100

    try:
        perf_score = df['diff'].mean()
        perf_dev = df['diff'].std()

    except Exception as e:
        print(f'/!\\ Warning exception {e} occurred')

    print()
    print(f'Statistics               |     Value | Pass |')
    print(f'-------------------------|-----------|------|')
    print(f'Bench Passes             :           | {len(df["sd"]) == 19}')
    print(f'Deviation Quantile (80%) : {sd_quantile:+.4f} % | {sd_quantile < 5} |')

    if False:
        print(f'Deviation sd             : {      sd_sd:+.4f} % | {sd_sd < 5} |')

    if perf_score:
        print(f'Performance              : {perf_score:+.4f} % | {perf_score >= 0} ')

        if False:
            print(f'Performance sd           : {perf_dev:+.4f} % | {perf_score >= 0} ')

    print('--')


if __name__ == '__main__':
    main()

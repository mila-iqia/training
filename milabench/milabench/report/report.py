from hrepr import HTML
from collections import defaultdict
import json
import os
import numpy as np
import glob
import pandas as pd
from pandas import DataFrame, Series
import math
from itertools import chain


H = HTML()


_exclude = {
    'convnet_distributed',
    'convnet_distributed_fp16',
    'dcgan_all',
}


def load_comparison_data2(gpu_model):
    import os
    from milabench import report
    dirname = os.path.dirname(report.__file__)
    return pd.read_csv(f'{dirname}/target_{gpu_model}.csv',
                       index_col='bench')


def read_results(report_name):
    str_report = open(report_name, 'r').read().strip()[:-1]
    if not str_report.endswith(']'):
        str_report += ']'
    return json.loads(str_report)


def extract_reports(report_folder, report_names='baselines'):
    filenames = chain(
        glob.glob(f'{report_folder}/**/{report_names}_*.json'),
        glob.glob(f'{report_folder}/{report_names}_*.json')
    )
    reports = defaultdict(list)
    for filename in filenames:
        data = read_results(filename)
        for entry in data:
            if entry['name'] in _exclude:
                continue
            entry['__path__'] = filename
            reports[entry['name']].append(entry)

    return reports


def _report_nans(reports):
    nans = []
    for name, entries in reports.items():
        for entry in entries:
            if any(math.isnan(x) for x in entry['batch_loss']):
                nans.append(entry)
    return nans


def _report_pergpu(baselines, reports, measure='mean'):
    results = defaultdict(lambda: defaultdict(list))
    all_devices = set()
    for name, entries in reports.items():
        for entry in entries:
            devices = [int(device_id) for device_id in entry['vcd'].split(',')]
            if len(devices) == 1:
                device, = devices
                all_devices.add(device)
                results[name][device].append(entry['train_item']['avg'])

    all_devices = list(sorted(all_devices))

    results = {
        name: {device: getattr(Series(data), measure)()
               for device, data in device_results.items()}
        for name, device_results in results.items()
    }

    df = DataFrame(results).transpose()
    df = df.reindex(columns=all_devices)

    maxes = df.loc[:, all_devices].max(axis=1).transpose()
    df = (df.transpose() / maxes).transpose()

    return df


_table_style = H.style("""
body {
    font-family: monospace;
}
td, th {
    text-align: right;
    min-width: 75px;
}
.result-PASS {
    color: green;
    font-weight: bold;
}
.result-FAIL {
    color: red;
    font-weight: bold;
}
""")


def _style(df):

    def _redgreen(value):
        return 'color: green' if value else 'color: red'

    def _gpu_pct(value):
        if value >= 0.9:
            color = '#080'
        elif value >= 0.8:
            color = '#880'
        elif value >= 0.7:
            color = '#F80'
        else:
            color = '#F00'
        return f'color: {color}'

    def _perf(values):
        return (values >= df['perf_tgt']).map(_redgreen)

    def _sd(values):
        return (values <= df['sd%_tgt']).map(_redgreen)

    # Text formatting
    sty = df.style
    sty = sty.format(_formatters)

    # Format GPU efficiency map columns
    gpu_columns = set(range(16)) & set(df.columns)
    sty = sty.applymap(_gpu_pct, subset=list(gpu_columns))

    # Format perf column
    if 'perf' in df.columns and 'perf_tgt' in df.columns:
        sty = sty.apply(_perf, subset=['perf'])
        sty = sty.applymap(
            lambda x: 'font-weight: bold' if x >= 1 else '',
            subset=['perf']
        )

    # Format sd% column
    if 'sd%' in df.columns and 'sd%_tgt' in df.columns:
        sty = sty.apply(_sd, subset=['sd%'])

    # Format pass/fail column
    for col in ['pass', 'sd%_pass', 'perf_pass']:
        if col in df.columns:
            sty = sty.applymap(_redgreen, subset=[col])

    return sty


def _display_title(file, title, stdout_display=True):
    if stdout_display:
        print()
        print('=' * len(title))
        print(title)
        print('=' * len(title))
        print()

    if file:
        print(H.h2(title), file=file)


def _display_table(file, title, table, stdout_display=True):
    _display_title(file, title, stdout_display=stdout_display)

    if stdout_display:
        print(table.to_string(formatters=_formatters))

    if file:
        sty = _style(table)
        print(sty._repr_html_(), file=file)


def _report_global(baselines, reports):
    df = DataFrame(
        {
            name: {
                'n': len(entries),
                'mean': Series(x['train_item']['avg'] for x in entries).mean(),
                'sd': Series(x['train_item']['avg'] for x in entries).std(),
            }
            for name, entries in reports.items()
        }
    ).transpose()
    df['target'] = baselines['target']
    df['perf_tgt'] = baselines['perf_target']
    df['sd%_tgt'] = baselines['sd_target']
    df['perf'] = df['mean'] / df['target']
    df['sd%'] = df['sd'] / df['mean']
    df['perf_pass'] = df['perf'] >= df['perf_tgt']
    df['sd%_pass'] = df['sd%'] <= df['sd%_tgt']
    df['pass'] = df['perf_pass'] & df['sd%_pass']
    return df


def passfail(x):
    return 'PASS' if x else 'FAIL'


_formatters = {
    'n': '{:.0f}'.format,
    'target': '{:10.2f}'.format,
    'sd%_tgt': '{:10.0%}'.format,
    'mean': '{:10.2f}'.format,
    'sd': '{:10.2f}'.format,
    'perf': '{:10.2f}'.format,
    'sd%': '{:10.1%}'.format,
    'pass': passfail,
    'sd%_pass': passfail,
    'perf_pass': passfail,
    0: '{:.0%}'.format,
    1: '{:.0%}'.format,
    2: '{:.0%}'.format,
    3: '{:.0%}'.format,
    4: '{:.0%}'.format,
    5: '{:.0%}'.format,
    6: '{:.0%}'.format,
    7: '{:.0%}'.format,
    8: '{:.0%}'.format,
    9: '{:.0%}'.format,
    10: '{:.0%}'.format,
    11: '{:.0%}'.format,
    12: '{:.0%}'.format,
    13: '{:.0%}'.format,
    14: '{:.0%}'.format,
    15: '{:.0%}'.format,
}


def _format_df(df):
    return df.to_string(formatters=_formatters)


def main():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--reports',
                        type=str,
                        help=('folder containing all the reports:'
                              ' reports/folder/jobs.json'))
    parser.add_argument('--jobs',
                        type=str,
                        default='fast',
                        help='name of the report to load')
    parser.add_argument('--gpu-model',
                        choices=['V100', 'MI50', 'RTX'],
                        help='GPU model to compare against')
    parser.add_argument('--html', type=str, default='',
                        help='file in which to save the report')
    parser.add_argument('--title', type=str, default='',
                        help='title for the report')

    args = parser.parse_args()

    reports = extract_reports(args.reports, args.jobs)

    # Nan check
    nans = _report_nans(reports)
    for nan in nans:
        print(f'nan found in batch loss for test "{name}"',
              f'in file {entry["__path__"]}')

    baselines = load_comparison_data2(args.gpu_model)

    outd = True
    title = args.title or args.html.replace('.html', '')

    html = args.html and open(args.html, 'w')
    print(f'<html><head><title>{title}</title></head><body>', file=html)
    print(_table_style, file=html)

    _display_title(
        title=f'{title} ({args.reports})',
        file=html,
        stdout_display=outd
    )

    df_global = _report_global(baselines, reports)
    df = df_global.reindex(
        columns=(
            'n',
            'target',
            'perf_tgt',
            'sd%_tgt',
            'mean',
            # 'sd',
            'perf',
            'perf_pass',
            'sd%',
            'sd%_pass',
            # 'pass',
        )
    )
    _display_table(
        title=f'Results',
        table=df,
        file=html,
        stdout_display=outd,
    )

    _display_title(title='Global performance', file=html, stdout_display=outd)

    try:
        df_perf = df.drop('scaling')
    except KeyError:
        df_perf = df
    perf = (df_perf['perf'].prod()) ** (1/df_perf['perf'].count())
    minreq = 0.9
    success = perf > minreq
    grade = (df_global['perf_pass'].sum()
             + df_global['sd%_pass'].sum()
             + success)
    expected = 63
    grade_success = grade == expected

    if outd:
        print(f'Mean performance (geometric mean): {perf:.2f}')
        print(f'Minimum required:                  {minreq:.2f}')
        print(f'Mean performance (pass or fail):   {passfail(success)}')
        print(f'Grade:                             {grade}/{expected}')
        print(f'Pass or fail (final):              {passfail(grade_success)}')

    if html:
        tb = H.table(
            H.tr(
                H.th('Mean performance (geometric mean)'),
                H.td(f'{perf:.2f}'),
            ),
            H.tr(
                H.th('Minimum required'),
                H.td(f'{minreq:.2f}'),
            ),
            H.tr(
                H.th('Mean performance (pass or fail)'),
                H.td[f'result-{passfail(success)}'](passfail(success)),
            ),
            H.tr(
                H.th('Grade'),
                H.td[f'result-{passfail(grade_success)}'](
                    f'{grade}/{expected}'
                ),
            ),
            H.tr(
                H.th('Pass or fail (final)'),
                H.td[f'result-{passfail(grade_success)}'](
                    passfail(grade_success)
                ),
            ),
        )
        print(tb, file=html)

    for measure in ['mean', 'min', 'max']:
        df = _report_pergpu(baselines, reports, measure=measure)
        _display_table(
            title=f'Relative GPU performance ({measure})',
            table=df,
            file=html,
            stdout_display=outd,
        )

    html.write('</body></html>')


if __name__ == '__main__':
    main()

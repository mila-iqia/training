from typing import *
import os.path
from benchutils.statstream import StatStream


class UnEvenTable(Exception):
    def __init__(self, message):
        self.message = message


class PrintTable:
    """
        Generate a markdown/csv table
    """

    def __init__(self, cols, data):
        self.format = {
            float: '.4f',
            int: '4d'
        }

        self.columns = cols
        self.data = data
        self.col_num = len(self.columns)
        self.check_size()

        self.col_size = self.compute_col_size()
        self.row_size = self.compute_row_size()
        self.print_fun = print

    def check_size(self):
        for row_id, row in enumerate(self.data):
            if len(row) != self.col_num:
                print(row)
                raise UnEvenTable('Row ({}) has not the correct number of columns {} != {}'
                                  .format(row_id, len(row), self.col_num))

    def compute_val_size(self, value: Any) -> int:
        return len(self.format_cell(value)) + 2

    def format_cell(self, value: Any) -> str:
        fmt = self.format.get(type(value))
        if fmt is not None:
            return ('{:' + fmt + '}').format(value)
        return str(value)

    def compute_col_size(self):
        col_sizes = [float('-inf')] * self.col_num

        def max_col(_, col_id, val):
            col_sizes[col_id] = max(col_sizes[col_id], self.compute_val_size(val))

        self.foreach(max_col, header=True)
        return col_sizes

    def compute_row_size(self):
        return sum(self.col_size)

    def aligned_print(self, str, length, align, end=''):
        missing = max(length - len(str), 0)
        if align == 'left':
            self.print_fun(str + ' ' * missing + end, end='')

        if align == 'right':
            self.print_fun(' ' * missing + str + end, end='')

        if align == 'center':
            r = missing // 2
            l = r + missing % 2
            self.print_fun(' ' * l + str + ' ' * r + end, end='')

        if align is None:
            self.print_fun(str + end, end='')

    def print(self, header=True, mode='csv'):
        impl = {
            'csv': self.csv_print,
            'md': self.markdown_print
        }

        return impl[mode](header)

    def csv_print(self, header):
        def simple(rowd_id, col_id, val):
            end = ','
            if col_id == len(self.col_size) - 1:
                end = ''

            self.aligned_print(' ' + self.format_cell(val) + ' ', self.col_size[col_id], 'right', end=end)

        self.foreach(simple, header, beg_row=None, end_row=lambda x: self.print_fun(), after_header=None)

    def markdown_print(self, header):
        def simple(rowd_id, col_id, val):
            self.aligned_print(' ' + self.format_cell(val) + ' ', self.col_size[col_id], 'right', end='|')

        def md_header_separator():
            cols = ['-' * (int(size) - 1) + ':' for size in self.col_size]

            self.print_fun('|' + '|'.join(cols) + '|')

        self.foreach(simple, header,
                     beg_row=lambda x: self.print_fun('|', end=''),
                     end_row=lambda x: self.print_fun(), after_header=md_header_separator)

    def foreach(self, fun, header, beg_row=None, end_row=None, after_header=None):
        if header:
            if beg_row is not None:
                beg_row(-1)

            for col_id, header in enumerate(self.columns):
                fun(-1, col_id, header)

            if end_row is not None:
                end_row(-1)

            if after_header is not None:
                after_header()

        for row_id, row in enumerate(self.data):
            if beg_row is not None:
                beg_row(row_id)

            for col_id, col in enumerate(row):
                fun(row_id, col_id, col)

            if end_row is not None:
                end_row(row_id)


def print_table(cols, data, filename=None, skip_header=True):
    report = PrintTable(cols, data)
    report.print()

    if filename is not None:
        if skip_header:
            header = not os.path.exists(filename)
        else:
            header = True

        append_file = open(filename, 'a')

        def new_print(self='', *args, sep=' ', end='\n', file=None):
            print(self, *args, sep=sep, end=end, file=append_file)

        report.print_fun = new_print
        report.print(header)

        append_file.close()


def print_stat_streams(names: List[str], stats: List[StatStream], additional_names=[], additional_cols=[], file_name: str=None, skip_header=True):
    cols = ['Name', 'Average', 'SD', 'Min', 'Max', 'Total', 'Count'] + additional_names
    data = [[name, stat.avg, stat.sd, stat.min, stat.max, stat.total, stat.count] + additional_cols for name, stat in zip(names, stats)]
    print_table(cols, data, file_name, skip_header)


if __name__ == '__main__':
    PrintTable(['A', 'B', 'C'], [
        ['qwerty', 1.23456789, 4567890],
        ['qwerty', 1.23, 4567890],
        ['qwerty', 4567891.23456789, 4567890]
    ]).print()

    print_table(['A', 'B', 'C'], [
        ['qwerty', 1.23456789, 4567890],
        ['qwerty', 1.23, 4567890],
        ['qwerty', 4567891.23456789, 4567890]
    ], 'test.txt')

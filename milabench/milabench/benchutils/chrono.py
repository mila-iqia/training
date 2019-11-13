import time
import json

from .statstream import StatStream
from .report import print_table

from math import sqrt
from math import log10
from typing import List, Dict
from typing import Callable


def chrono(func: Callable):
    def chrono_decorator(*args, **kwargs):
        start = time.time()
        value = func(*args, **kwargs)
        t = time.time() - start
        print('{:>30} ran in {:10.4f} s'.format(func.__name__, t))
        return value
    return chrono_decorator


class _DummyContext:
    def __init__(self, **args):
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class ChronoContext:
    """
        sync is a function that can be set to make the timer wait before ending.
        This is useful when timing async calls like cuda calls
    """
    def __init__(self, name, stream: StatStream, sync: Callable, parent, verbose=False, endline='\n'):
        self.name = name
        self.stream = stream
        self.start = 0
        self.sync = sync
        self.parent = parent
        self.verbose = verbose
        self.newline = endline

    def __enter__(self):
        # Sync before starting timer to make sure previous work is not timed as well
        self.depth = self.parent.depth
        self.parent.depth += 1
        self.sync()

        #if self.verbose:
        #    print(f'{" " * self.depth * 2} [{self.depth:3d}] >  {self.name}', end='')

        self.start = time.time()
        return self.stream

    def __exit__(self, exception_type, exc_val, traceback):
        # Sync before ending timer to make sure all the work is accounted for
        self.sync()
        self.end = time.time()

        self.parent.depth -= 1
        if exception_type is None:
            self.stream.update(self.end - self.start)

        if self.verbose:
            print(
                f'{self.newline}{" " * self.depth * 2} [{self.depth:3d}] <  {self.name:>30}: (obs: {self.stream.val:8.4f} s, '
                f'avg: {self.stream.avg:8.4f}, '
                f'cnt: {self.stream.count})',
                end=''
            )

    @property
    def count(self):
        return self.stream.current_count


class MultiStageChrono:
    def __init__(self, skip_obs=10, sync=None, disabled=False, name=None):
        self.chronos = {}
        self.skip_obs = skip_obs
        self.sync = sync
        self.name = name
        self.disabled = disabled
        self.depth = 0
        if sync is None:
            self.sync = lambda: None

    def time(self, name, skip_obs=None, **kwargs):
        if self.disabled:
            return _DummyContext()

        # if self.name is not None:
        #    name = '{}.{}'.format(self.name, name)

        val = self.chronos.get(name)

        if val is None:
            val = StatStream(self.skip_obs)
            if skip_obs is not None:
                val = StatStream(skip_obs)
            self.chronos[name] = val

        # inherit sync from parent
        if kwargs.get('sync') is None:
            kwargs['sync'] = self.sync

        return ChronoContext(name, val, parent=self, **kwargs)

    def make_table(self, common: List = None, transform=None):
        common = common or []
        table = []

        for i, (name, stream) in enumerate(self.chronos.items()):
            table.append([name] + stream.to_array(transform) + common)

        return table

    def report(self, *args, format='csv', **kwargs):
        if format == 'csv':
            return self.report_csv(*args, **kwargs)
        print(self.to_json(*args, **kwargs))

    def report_csv(self, speed=False, size=1, file_name=None, common: Dict[str, str] = None, skip_header=False):
        if self.disabled:
            return

        common = common or {}

        # split map in two
        items = list(common.items())

        common_header = list(map(lambda item: item[0], items))
        common = list(map(lambda item: item[1], items))

        header = ['Stage', 'Average', 'Deviation', 'Min', 'Max', 'count']
        header.extend(common_header)

        table = self.make_table(common, lambda x: size / x) if speed else self.make_table(common)
        print_table(header, table, file_name, skip_header)

    def to_dict(self, base=None):
        items = base
        if items is None:
            items = {}

        if self.name is not None:
            items['name'] = self.name

        for key, stream in self.chronos.items():
            items[key] = stream.to_dict()

        return items

    def to_json(self, base=None, *args, **kwargs):
        if 'indent' not in kwargs:
            kwargs['indent'] = '  '
        return json.dumps(self.to_dict(base), *args, **kwargs)


def time_this(chrono, *cargs, **ckwargs):
    def toplevel_decorator(fun):
        def wrapper(*args, **kwargs):
            with chrono.time(fun.__name__, *cargs, **ckwargs):
                return fun(*args, **kwargs)

        return wrapper
    return toplevel_decorator


def estimated_time_to_arrival(i: int, n: int, timer: StatStream):
    # return ETA and +/- offset
    avg = timer.avg
    if avg == 0:
        avg = timer.val

    eta = (n - i - 1) * avg
    return eta, 2.95 * timer.sd * sqrt((n - i - 1))


def get_div_fmt(val: float):
    div, fmt = 60, 'min'
    if val < div:
        div = 1
        fmt = 's'
    return div, fmt


def show_eta(i: int, n: int, timer: StatStream, end='\n'):
    eta, offset = estimated_time_to_arrival(i, n, timer)
    size = int(log10(n) + 1)

    div, fmt = get_div_fmt(eta)

    eta = f'{eta / div:6.2f} {fmt}'

    div, fmt = get_div_fmt(offset)
    conf = f'{offset / sqrt(div):6.2f} {fmt}'

    total = ''
    if timer.avg != 0:
        t = timer.avg * (timer.count + timer.drop_obs)
        div, fmt = get_div_fmt(t)
        total = f' | Total {t / div:6.2f} {fmt}'

    print(f'[{i:{size}d}/{n:{size}d}] Remaining {eta} +/- {conf}{total}', end=end)


if __name__ == '__main__':

    chrono = MultiStageChrono(0, disabled=False)

    with chrono.time('all', verbose=True):
        for i in range(0, 10):

            with chrono.time('forward_back', verbose=True) as timer:
                with chrono.time('forward', verbose=True):
                    time.sleep(1)

                    if i % 2 == 0:
                        time.sleep(0.25)

                with chrono.time('backward', verbose=True, skip_obs=3):
                    time.sleep(1)

                    if i % 2 == 0:
                        time.sleep(0.25)

        show_eta(i, 10, timer)

    print()
    chrono.report()
    print(chrono.to_json(base={'main': 1}, indent='   '))

    print()
    chrono.report(format='json')

    @time_this(chrono, verbose=True)
    def test():
        time.sleep(1)

    test()



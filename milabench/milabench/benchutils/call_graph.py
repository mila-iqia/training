from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput
from pycallgraph import Config
from pycallgraph import GlobbingFilter

NO_CALL_GRAPHS = True


def make_callgraph(name: str, id: str, dry_run=NO_CALL_GRAPHS) -> 'PyCallGraph':
    """
    :param name: file name used to generate the call graph
    :param id:  if of the image
    :param dry_run: if True do not generate a call graph
    :return: a context manager
    """
    class DummyCtx:
        def __enter__(self):
            pass

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    if dry_run:
        return DummyCtx()

    config = Config()
    config.trace_filter = GlobbingFilter(exclude=[
        'pycallgraph.*',
        'tornado*',
        '*SynchronizedStatStreamStruct*'
    ])
    output = GraphvizOutput(output_file='call_graphs/{}_{}.png'.format(name, id))
    return PyCallGraph(output=output, config=config)


#!/usr/bin/env python

from setuptools import setup


if __name__ == '__main__':
    setup(
        name='milabench',
        version='0.0.1',
        description='Implement a few utilities to help benchmark code',
        author=[
            'Pierre Delaunay',
            'Olivier Breuleux',
        ],
        packages=[
            'milabench',
        ],
        entry_points={
            'console_scripts': [
                'mlbench-report = milabench.report.report:main',
            ]
        },
        include_package_data=True,        
    )

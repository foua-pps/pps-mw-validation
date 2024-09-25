#!/usr/bin/env python3
from pathlib import Path
import re
import setuptools


here = Path(__file__).parent.absolute()
required = [
    r for r in (here / 'requirements.txt').read_text().splitlines()
    if '=' in r or "git" in r
]
version = re.findall(
    r'__version__ *= *[\'"]([^\'"]+)',
    (here / 'pps_mw_validation' / '__init__.py').read_text(encoding='utf-8')
)[-1]
long_description = """
    Package for NWCSAF/PPS-MW product validation.
    TODO: add more text
"""

setuptools.setup(
    name='pps-mw-validation',
    version=version,
    description='Package for NWCSAF/PPS-MW product validation.',
    author='Bengt Rydberg',
    author_email='bengt.rydberg@smhi.se',
    url='http://nwcsaf.org',
    long_description=long_description,
    license='GPL',
    packages=setuptools.find_packages(),
    python_requires='>=3.9, <4',
    install_requires=required,
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'resample=scripts.resample:cli',
            'collect=scripts.collect:cli',
            'compare=scripts.compare:cli',
            'prhl=scripts.prhl:cli',
        ],
    }
)

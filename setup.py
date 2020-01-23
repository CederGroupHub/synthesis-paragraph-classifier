import os

from setuptools import setup, find_packages

__author__ = 'Haoyan Huo'
__maintainer__ = 'Haoyan Huo'
__email__ = 'haoyan.huo@lbl.gov'

with open(os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'requirements.txt')) as f:
    requirements = list(filter(lambda x: x.strip(), f.readlines()))

if __name__ == '__main__':
    setup(
        name='synthesis_paragraph_classifier',
        description='Identify inorganic synthesis paragraphs (https://doi.org/10.1038/s41524-019-0204-1)',
        url='https://github.com/CederGroupHub/synthesis-paragraph-classifier',
        author='Haoyan Huo',
        author_email='haoyan.huo@lbl.gov',
        include_package_data=True,
        version='0.1.0',
        packages=find_packages(),
        install_requires=requirements,
    )

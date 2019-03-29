# -*- coding: utf-8 -*-
"""
Installs binaries:
    - ocropus-gpageseg-with-coords
    - ocrd-pc-seg-process
    - ocrd-pc-seg-single
"""
import codecs

from setuptools import setup, find_packages

setup(
    name='ocrd_pc_segmentation',
    version='0.1.0',
    description='pixel-classifier based page segmentation',
    long_description=codecs.open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    author='Alexander Gehrke, Christian Reul, Christoph Wick',
    author_email='alexander.gehrke@uni-wuerzburg.de, christian.reul@uni-wuerzburg.de, christoph.wick@uni-wuerzburg.de',
    url='https://github.com/ocr-d-modul-2-segmentierung/segmentation-runner',
    license='Apache License 2.0',
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=open("requirements.txt").read().split(),
    extras_require={
        'tf_cpu': ['page-segmentation[tf_cpu]>=0.0.1'],
        'tf_gpu': ['page-segmentation[tf_gpu]>=0.0.1'],
    },
    package_data={
        '': ['*.json', '*.yml', '*.yaml'],
    },
    entry_points={
        'console_scripts': [
            'ocrd-pc-seg-single=ocrd_pc_segmentation.pc_segmentation:main',
            'ocrd-pc-seg-process=ocrd_pc_segmentation.seg_process:main',
        ]
    },
    scripts=['ocrd_pc_segmentation/ocropus-gpageseg-with-coords'],
)

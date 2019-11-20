# -*- coding: utf-8 -*-
import codecs

from setuptools import setup, find_packages

setup(
    name='ocrd_pc_segmentation',
    version='0.1.3',
    description='pixel-classifier based page segmentation',
    long_description=codecs.open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    author='Alexander Gehrke, Christian Reul, Christoph Wick',
    author_email='alexander.gehrke@uni-wuerzburg.de, christian.reul@uni-wuerzburg.de, christoph.wick@uni-wuerzburg.de',
    url='https://github.com/ocr-d-modul-2-segmentierung/segmentation-runner',
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=open("requirements.txt").read().split(),
    extras_require={
        'tf_cpu': ['ocr4all_pixel_classifier[tf_cpu]>=0.0.1'],
        'tf_gpu': ['ocr4all_pixel_classifier[tf_gpu]>=0.0.1'],
    },
    package_data={
        '': ['*.json', '*.yml', '*.yaml'],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Image Recognition"

    ],
    entry_points={
        'console_scripts': [
            'ocrd-pc-segmentation=ocrd_pc_segmentation.cli:ocrd_pc_segmentation',
        ]
    },
    data_files=[('', ["requirements.txt"])],
    include_package_data=True,
)

from setuptools import setup

setup(
    name='AutoCATE',
    version='0.0.1',
    description='End-to-end, automated treatment effect estimation',
    url='https://github.com/toonvds/AutoCATE',
    author='Toon Vanderschueren',
    author_email='toon.vanderschueren@gmail.com',
    license='MIT',
    packages=['AutoCATE'],
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'scikit-learn',
        'causalml',
        'tqdm',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering',
    ],
)

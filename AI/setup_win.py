from setuptools import setup

setup(
    name='AI',
    version='1.0',
    py_modules=['AI','sgtlink'],
    install_requires=[
        'Click',
        'numpy',
        'sigopt',
        'slackclient',
        'scikit-image',
        'scipy',
        'tqdm',
        'visdom'
    ],
    entry_points='''
        [console_scripts]
        AI=AI:cli
        sgtlink=sgtlink:eval
    ''',
)

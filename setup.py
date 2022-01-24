from setuptools import setup, find_packages

setup(
    name='dl_toolbox',
    version='1.0.0',
    author='Pierre Fournier',
    author_email='pierre.fournier@onera.fr',
    description='Deep learning toolbox',
    packages=find_packages(where='dl_toolbox'),
    install_requires=[
        'segmentation-models-pytorch'
        ]
)

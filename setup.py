from setuptools import setup, find_packages # type: ignore

setup(
    name='Entanglement_Routing',
    version='0.1.0',
    author='David Fainsin',
    author_email='david.fainsin@lkb.upmc.fr',
    description='See Arxiv',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        # List your project dependencies here
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
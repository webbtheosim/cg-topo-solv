from setuptools import setup, find_packages

setup(
    name='cg_topo_solv',
    version='0.0.1',
    description="Data-driven inverse design of polymer rheology using graph-based generative modeling, active learning, and coarse-grained simulations with topology and solvophobicity control.",
    author='Shengli (Bruce) Jiang',
    author_email='sj0161@princeton.com',
    url='https://github.com/webbtheosim/cg-topo-solv',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.0',
        'matplotlib>=3.0',
    ],
)

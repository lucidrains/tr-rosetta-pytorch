from setuptools import setup, find_packages

setup(
  name = 'tr-rosetta-pytorch',
  packages = find_packages(),
  include_package_data = True,
  entry_points={
    'console_scripts': [
      'tr_rosetta = tr_rosetta_pytorch.cli:predict',
    ],
  },
  version = '0.0.2',
  license='MIT',
  description = 'trRosetta - Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/tr-rosetta-pytorch',
  keywords = [
    'artificial intelligence',
    'protein folding',
    'protein design'
  ],
  install_requires=[
    'einops>=0.3',
    'fire',
    'numpy',
    'torch>=1.6'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)

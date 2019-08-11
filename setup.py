import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(name='pysaber',
      version='0.1.5',
      description='Python package that implements a systems approach to blur estimation and reduction (SABER)',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/sabersw',
      author='K. Aditya Mohan',
      author_email='mohan3@llnl.gov',
      license='MIT',
      packages=setuptools.find_packages(),
      install_requires=['numpy','pyyaml','scipy','scikit-image','scikit-learn','matplotlib'],
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
      ],
      )

from setuptools import setup, find_packages


def readme():
    with open('README.md') as f:
        return f.read()


setup(name='preprocess_functions',
      version='0.0.1',
      description='Pre-process functions for data analysis in Python',
      long_description=readme(),
      classifiers=[
          'Programming Language :: Python',
          'Programming Language :: Python :: 3.6',
      ],
      keywords=['data-manipulation', 'data-preprocessing',
                'pandas', 'Python'],
      url='https://github.com/mloning/preprocess_functions',
      author='Markus Löning',
      author_email='markus.loning.17@ucl.ac.uk',
      license='MIT',
      packages=find_packages(exclude=['*.tests', '*.tests.*', 'tests.*',
                                      'tests', 'test_*']),
      install_requires=[
        'numpy', 'pandas'
      ],
      test_suite='nose.collector',
      tests_require=['nose'],
      include_package_data=True,
      zip_safe=False)
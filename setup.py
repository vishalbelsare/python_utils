from setuptools import setup, find_packages


def readme():
    with open('README.md') as f:
        return f.read()


setup(name='python_utils',
      version='0.0.1',
      description='Collection of useful functions for data analysis in Python',
      long_description=readme(),
      classifiers=[
          'Programming Language :: Python',
          'Programming Language :: Python :: 3.6'
      ],
      keywords=['data-manipulation', 'data-preprocessing',
                'pandas', 'Python'],
      url='https://github.com/mloning/python_utils',
      author='mloning',
      author_email='markus.loning.17@ucl.ac.uk',
      license='MIT',
      packages=find_packages(exclude=['*.tests', '*.tests.*', 'tests.*',
                                      'tests', 'test_*']),
      python_requires='>=3, <4',
      install_requires=[
          'numpy>=1.13',
          'pandas>=0.22'
      ],
      test_suite='nose.collector',
      tests_require=['nose'],
      include_package_data=True,
      zip_safe=False)

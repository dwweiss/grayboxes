from setuptools import setup

setup(name='grayBoxes',
      version='0.1',
      description='Gray box modelling',
      url='https://github.com/dwweiss/grayBoxes',
      keywords='modelling, gray box, grey box, hybrid',
      author='Dietmar Wilhelm Weiss',
      license='GLGP 3.0',
      platforms=['Linux', 'Windows'],
      packages=['src'],
      include_package_data=True,
      install_requires=['numpy', 'matplotlib', 'pandas', 'scipy', 'neurolab',
                        'modestga'],
      classifiers=['Programming Language :: Python :: 3',
                   'Topic :: Scientific/Engineering',
                   'License :: OSI Approved :: GLGP 3.0 License']
      )

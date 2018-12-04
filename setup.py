from setuptools import setup, find_packages

setup(name='OOPT-Raman',
      version='0.1dev',
      description='A Raman solver for optical fiber communication systems.',
      author='Alessio Ferrari',
      author_email='alessio.ferrari@polito.it',
      license='BSD 3-Clause License',
      packages=find_packages(exclude=['tests']),
      install_requires=list(open('requirements.txt')),
      long_description=open('README.md').read(),
  )

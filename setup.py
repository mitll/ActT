from setuptools import find_packages, setup

setup(name='ActT',
      version='0.6',
      description='Active Testing',
      url='',
      author='',
      author_email='none',
      license='BSD',
      packages=find_packages(),
      install_requires=[
        'numpy', 'pandas', 'matplotlib','scipy','scikit-learn','opencv-python',
        'statswag','tensorflow'
      ],
      zip_safe=True)

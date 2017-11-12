from setuptools import setup

setup(
    name='chimera',
    version='1.0.0',
    description='Chimera synthesis',
    author='Stephen Lumenta, Bertrand Delgutte',
    license='MIT',
    author_email='stephen.lumenta@gmail.com',
    packages=['chimera'],
    install_requires=[
        'numpy>=1.13.0',
        'scipy>=1.0.0',
    ],
)

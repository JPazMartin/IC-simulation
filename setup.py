from setuptools import setup

setup(
    name         = 'ICSimulation',
    version      = '1.0.0',
    author       = "José Paz-Martín",
    author_email = "jose.martin@usc.es",
    description  = "Code for simulating parallel-plate, cylindrical and spherical ionization chambers",
    keywords     = "recombination ionization chambers",
    packages     = ["ICSimulation"],
    data_files   = [('data', ['data/dataElectrons.txt'])]
)
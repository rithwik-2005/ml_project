from setuptools import setup,find_packages

def read_requirements(file_path):
    with open(file_path) as file:
        lines = file.read().splitlines()
        return [line for line in lines if line.strip() and line.strip() != "-e ."]

setup(
    name="ml_project",
    version="0.0.1",
    author="Rithwik",
    author_email="rithwikvasa@gmail.com",
    packages=find_packages(),
    install_requires=read_requirements("requirements.txt")
)

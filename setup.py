from setuptools import setup,find_packages


def read_requirements(file_path):
    with open(file_path) as file:
        lines=file.read().splitlines()
        return [line for line in lines if line !="-e ."]
    

setup(
    name="ml_project",
    version="0.0.1",
    author="rithwik",
    author_email="rithwikvasa@gamil.com",
    packages=find_packages(),
    install_requires=read_requirements("requirements.txt")

)
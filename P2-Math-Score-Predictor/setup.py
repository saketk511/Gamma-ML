from setuptools import find_packages, setup  # Import necessary functions from setuptools
from typing import List  # Import List type for type hinting

# Constant to identify and remove the editable install entry '-e .' from requirements
HYPEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    '''
    This function reads the requirements.txt file and returns a list of requirements,
    excluding any instances of '-e .'
    '''
    requirements = []  # Initialize an empty list to store requirements
    with open(file_path) as file_obj:  # Open the requirements file
        requirements = file_obj.readlines()  # Read all lines from the file
        # Remove newline characters from each requirement
        requirements = [req.replace("\n", "") for req in requirements]

        # Remove the '-e .' entry if it exists in the requirements list
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
    return requirements  # Return the list of processed requirements

# Metadata and configuration for the package setup
setup(
    name='mlproject',  # Name of the package
    version='0.0.1',  # Initial version of the package
    author='Saket',  # Author's name
    author_email='saketk101@gmail.com',  # Author's email address
    packages=find_packages(),  # Automatically find and include all packages
    install_requires=get_requirements('requirements.txt')  # Install dependencies from the requirements.txt file
)

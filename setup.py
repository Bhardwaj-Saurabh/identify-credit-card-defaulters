from setuptools import find_packages, setup
from typing import List

# funtion to add requirement to the list
def get_requirements()->List[str]:
    '''
    Returns the list of requirements
    '''
    requirement_list:List[str] = []
    with open('requirements.txt', 'r') as r:
        requirements = r.read()

    for requirement in requirements.split('\n'):
        requirement_list.append(requirement)

    return requirement_list

# set up for the sensor detector
setup(
    name='credit card defaul prediction',
    version='0.0.1',
    author='saurabh bhardwaj',
    author_email='aryan.saurabhbhardwaj@gmail.com',
    packages=find_packages(),
    install_required=get_requirements()
)
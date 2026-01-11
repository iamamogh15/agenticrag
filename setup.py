from setuptools import setup, find_packages
from typing import List

hypen = "-e ."

def get_packages() -> List:
    #this function will give list of packages
    packagesList:List[str] = [ ]
    try:
        with open("packages.txt","r") as file:
            #Reading lines inside packages.txt
            lines = file.readlines()
            for line in lines:
                packages = line.strip()
                #ignoring hypen
                if packages and packages != hypen:
                    packagesList.append(packages)
    except FileNotFoundError:
        print("packages.txt file not found")
    
    return packagesList

setup(
    name = "agenticrag",
    version = "0.0.1",
    author = "Amogh",
    author_email="amoghmath2000@gmail.com",
    packages=find_packages(),
    install_requires = get_packages()
)

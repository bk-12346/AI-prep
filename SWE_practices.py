##### SWE PRACTICES #####

### 1. MODULARITY ###
# -> divide code into shorter functional units
# -> 1. more readable   2. easier to fix when code breaks
# -> USE: packages, classes, methods

### 2. DOCUMENTATION ###
# -> show users how to use your project
# -> prevent confusion among collaborators
# -> USE: comments, doc-strings, self-documenting code

### 3. AUTOMATED TESTING ###
# -> save time
# -> fix bugs
# -> run testa anytime/anywhere

### PACKAGES and PyPi ###
## 1. PIP
# -> use pip to install modular packages
# pip install numpy
# -> pip will also install the package dependencies as long as they are on PyPi

## 2. HELP()
# -> use help() to view a functions documentation
# help(numpy.busday_count)
# -> can also call help() on the package
# help(np)

### CONVENTIONS ###
# unwritten rules
# PEP 8 - style-guide for python code
# -> pycodestyle package
# tells us about the violations in the code + the info for where we need to fix the issue
# pip install pycodestyle
# pycodestyle dictto_array.py
# ->>> output shows file_name:line_number:column_number: error_code error_description

# Import needed package
from multiprocessing import Value
import matplotlib
import pycodestyle

# Create a StyleGuide instance
style_checker = pycodestyle.StyleGuide()

# Run PEP 8 check on multiple files
result = style_checker.check_files(['nay_pep8.py', 'yay_pep8.py'])

# Print result of PEP 8 style check
print(result.messages)

### WRITING A PACKAGE ###
## PACKAGE STRUCTURE
# 2 elements: 1. directory 2. python file
# -> name of the directory is the name of the package
# -> short, all lower-case names
# -> file name should always be __init__.py
# -> this filename let's python know that this is a package

# work_dir
#   |_>  my_script.py
#   |_> package_name
#           |_> __init__.py

## IMPORTING a LOCAL PACKAGE
import my_package

### FUNCTIONALITY TO PACKAGES ###

# work_dir
#   |_>  my_script.py
#   |_> package_name
#           |_> __init__.py
#           |_> utils.py

## Adding Functionality
# define a function in utils.py
def we_need_to_talk(break_up=False):
    """Helper for communicating with significant other"""
    if break_up:
        print("Its not you, its me...")
    else:
        print("I <3 u!")

# -> utils file is a submodule
# -> can be imported with .func_name()

# so in my_script.py
import my_package.utils

my_package.utils.we_need_to_talk(break_up=False)

## Importing Functionality with __init.py__
# -> we can use the package's init file to make the utils function more easily accessible by the user

from .utils import we_need_to_talk
import my_package

my_package.we_need_to_talk(break_up=False)

## Extending Package Structure
# ->> only import the packages' keyy functionality to the init file to make it directly and easily accessible
# ->> we can also build a package inside a package ->> a sub-package

### PORTABILITY ###
# setup.py and requirements.txt are used
# -> these files list information about what dependencies we have used + allows us to describe the package with additional metadata

# work_dir
#   |_>  my_script.py
#   |_> setup.py
#   |_> requirements.txt
#   |_> package_name
#           |_> __init__.py
#           |_> utils.py

## Requirements.txt 
# -> different ways to add requirements
matplotlib
numpy==1.15.4
pycodestyle>=2.4.0

# to install the requirements file use in terminal
# pip install -r requirements.txt

## Setup.py
# -> tells pip how to install the actual package
# -> this info is used by PyPi if we decide to publish

from setuptools import setup

setup(
    name='my_package',
    version='0.0.1',
    description='An example for a package',
    author='BK',
    author_email='bk@bk.com',
    packages=['my_package'],    # lists the locations for all the init files in the package
    install_requires=[          # can differ from requirements.txt if we want to specify where pip should download packages from
        'matplotlib',
        'numpy==1.15.4',
        'pycodestyle>=2.4.0'
    ]
)

### ADDING CLASSES to PACKAGE ###
# -> OOP to write modular code
## CLASS
# -> class names should never have underscores
# ->  __init__ ->> initializes everything when a user wants to leverage the class

## SELF
# a way to refer to the class instance even though we don't know what the user is actually going to name their instance
# we don't need to pass a value to the self argument, it happens automatically behind the scenes

class MyClass:
    """
    A minimal class example

    :param value: value to set the ``attribute`` attribute
    :ivar attribute: contains the contents of ``value`` passed in init
    """

    # Method to create a new instance of MyClass
    def __init__(self, value):
        # Define attribute with the contents of the value param
        self.attribute = value

# -> add it to the init.py file to make it easily accessible
from .my_class import MyClass   # in work_dir/my_package/__init__.py

## CREATE AN INSTANCE OF CLASS
# class MyClass like a function and supply a string to the value parameter
# calling the class like this tells Python that we want to create an instance of our class by using the init method

import my_package

# Create instance of MyClass
my_instance = my_package.MyClass(Value='class attribute value')

# print class attribute value
print(my_instance.attribute)

# ->> right now class is just a container for the user provided text, this doesn't add much value
# we can add more methods and attributes besides init
# init is called when a user wants to create an instance of Document
# this would be a convinient location to put a tokenization step
# this will ensure that the doc is tokenized as soon as it is created

class Document:
    def __init__(self, text):
        self.text=text
        self.tokens = self._tokenize()

doc = Document('test_doc')
print(doc.tokens)

# .tokenize()
# -> comes from a premade function
# leading underscore used before tokenize because:
# ->> this method doesn't need to be public to the user because we want the tokenization to happen automatically without the user having to think about it

from .tokens_utils import tokenize

class Document:
    def __init__(self, text, token_regex=r'[a-zA-Z]+'):
        self.text = text
        self.tokens = self._tokenize()
    
    def _tokenize(self):    # only pass one parameter to the function, the prescribed self convention r=that will represent an instance of the Documnet object
        return tokenize(self.text)  # just call it oon the text attribute
    
 



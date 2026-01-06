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





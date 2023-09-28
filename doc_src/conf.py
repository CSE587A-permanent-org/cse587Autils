# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

project = 'cse587Autils'
copyright = '2023, Chase Mateusiak'
author = 'Michael Brent, Chase Mateusiak'
<<<<<<< HEAD
release = '3.2.1'
=======
release = '3.1.0'
>>>>>>> 3300994804d25735b6d9c4fee0a0829792628a89

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.viewcode',
              'sphinx.ext.autosectionlabel',
              'sphinx.ext.doctest',
              'sphinx.ext.inheritance_diagram',
              'sphinx.ext.intersphinx',
              'nbsphinx']

exclude_patterns = ['build', '**.ipynb_checkpoints', '**/~', '**/.~']


intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}

# Automatically extract typehints when specified and place them in
# descriptions of the relevant function/method.
autodoc_typehints = "description"

# Don't show class signature with the class' name.
autodoc_class_signature = "separated"

templates_path = ['_templates']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
html_theme_path = ['.']
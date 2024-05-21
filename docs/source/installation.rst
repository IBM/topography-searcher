Installation
==============

**Requirements**

TopSearch requires a number of standard scientific Python packages e.g. numpy, scipy and networkx.
There are additional, more complex, requirements for some of the machine-learning potentials, 
but these are not included in the base installation and should be installed separately.

**Install with pip**

topsearch is available at PyPI, and we can install the latest version using pip::

    pip install topsearch

**Install from source**

If you would like to both use and develop topsearch then it is necessary to clone the
git repository. We can get the source repository with::

    git clone https://github.com/IBM/topography-searcher.git

Once you have the repository we can install all the necessary requirements using::

    cd topography-searcher
    python -m pip install .

And to allow modifications to change the enivornment immediately use editable mode via::

    python -m pip install -e .

**Test installation**

We can check that the installation has been successful by running the test framework.
The tests can be run by calling::
    pytest

For a successful installation we should see all tests pass.
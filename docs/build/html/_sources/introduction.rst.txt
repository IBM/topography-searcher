Introduction
===============

TopSearch is a Python package designed to make topographical analysis of surfaces straightforward. Topographical analysis
has many uses in chemical physics `[1] <https://doi.org/10.1146/annurev-physchem-050317-021219>`_ and machine 
learning `[2] <https://doi.org/10.1039/D3DD00204G>`_. The methodology we use to map the topography is the 
energy landscape framework `[3] <https://doi.org/10.1017/CBO9780511721724>`_. This methodology originated, 
and been very successful in, chemical physics, where it can efficiently
determine the properties and time-evolution of physical systems. However, the energy landscape framework can be applied to any 
continuous surface and, therefore, there have been many applications to machine-learning loss functions to aid understanding
of the resulting models.

The analysis relies upon finding stationary points, points on a surface at which the gradient vanishes. These stationary points are separated
into minima (all pertubations increase function value) and transition states (maximal in one direction and minimal in the remaining).
We encode the topography of a surface through a network of these stationary points. Each transition state connects two minima, which
are those between which it is a maximum. Therefore, we can easily represent the topography using a weighted graph in which
minima are nodes and transition states are edges. Such a graph can be used to compute all properties of a physical system, and 
reveal important features of machine learning models. We illustrate such a graph below

.. figure:: StationaryPointsExample.png
    :height: 400px
    :width: 600px

    Minima are denoted by green circles, transition states by red circles and the connections between transition states and minima
    by solid black lines.
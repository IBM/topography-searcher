# BayesOpt scripts

This folder contains scripts to peform and analyse Bayesian optimisation using the energy landscape framework. We provide examples for two common test functions: the three-dimensional Schwefel function and the six-hump Camel function.

* `acquisition_landscape` - Construct the upper confidence bound surface for a given dataset, map the surface into a weighted graph and provide visualisation of its topography.
* `serial_bayesopt` - Perform serial Bayesian optimisation for the six-hump camelback function. At each epoch we compute the acquisition function landscape, map its topography and plot its structure both directly and with alternative visualiation schemes.
* `batch_bayesopt` - Performs batch Bayesian optimisation for the six-hump camelback function. The landscape of each acquisition function surface is mapped same as in `serial_bayesopt`, but we additionally use the landscape to select a diverse batch of points at each epoch.

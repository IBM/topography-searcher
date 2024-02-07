## Jupyter notebooks

In this folder we store Jupyter notebooks that introduce three of the major applications of the energy landscape framework within the TopSearch package. Each contains a complete overview of the main tasks that are needed to map topographies within the given application. Moreover, for the bayesian optimisation and dataset roughness applications we show how to use topographical information to improve ML model performance and explainability.

* `bayesian_optimisation.ipynb` - A tutorial in how to use TopSearch to explore the structure of acquisition function surfaces in Bayesian optimisation, provide visualisations of these high-dimensional spaces, and use the topography to construct an efficient batch selection scheme.

* `dataset_roughness.ipynb` - A tutorial to illustrate the use of TopSearch in exploring the roughness, and modelability, of datasets. We show how a topographical mapping of the dataset can be used to estimate dataset roughness via a frustration metric. The roughness measure correlates strongly with model error and can be used to estimate the appropriateness of a given set of features for a regression task.

* `example_function.ipynb` - An initial tutorial showing much of the key functionality needed to explore high-dimensional surfaces. In this case we explore a common test function (the two-dimensional Schwefel function), which allows us to directly plot the surface and its visualisations to explain the methodology of the energy landscape framework.

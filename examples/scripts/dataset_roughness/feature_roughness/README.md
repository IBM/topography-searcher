# Feature roughness

Here we estimate the roughness of a chemical dataset using the TopSearch package. This example is similar to `custom_dataset_roughness`, but
we use two larger datasets containing alternative representations of molecules. We estimate the roughness of the datasets using the 
[frustration metric](https://doi.org/10.26434/chemrxiv-2023-0zx26), which correlates strongly with the achievable regression model accuracy.
We load two datasets composed of the same molecules, but with alternate features, and distinguish the quality of each particular
molecular representation in modelling chemical space.


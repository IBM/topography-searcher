# Roughness sampling

Given the netowrk of minima and transition states, we now wish sample from regions of high roughness, regions where the model performance begins to rapidly decrease, to produce a dataset that aims to correct failings of the current embedding. The
frustration is associated with vectors in the latent space and we transform this into a continuous roughness surface via a sum of weighted Gaussians placed on each vector.

We place a multivariate Gaussian at the centre of each vector from minimum to transition state. The height of the Gaussian is scaled to equal the roughness value associated with it at its maximum. The basis is shifted such that the first axis is
directed along the corresponding vector, with the covariance fitted such that the Gaussian reaches 10% of its original value by the position of the transition state and minimum that start and end the vector. In all orthogonal directions that
variance is chosen to be constant at 10−4. The roughness at a given point is the sum over all Gaussians. 

We use basin-hopping to minimise this surface, inverting the response so that minima correspond to regions of highest roughness. 
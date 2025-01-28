# Data generation

We begin by sampling the latent space. We use Latin Hypbercube sampling to create 5000 artificial latent vectors evenly distributed across the space. These will form the features of our landsacpe.

To construct the response function for the landscape, we then pass each latent vector to the model's decoder. This returns SELFIES.  In order to get a metric for the variation of model response to latent vectors, we generate 10 neighbouring points in close proximity, and decode these. We then obtain the Morgan fingerprint for each neighbour and compute the Tanimoto similarity with the fingerprint of the latent vector. Finally, we take mean similarity over each neighbouring point to give the response value. 

In order to run this step, you must set up the SELFIES-TED code, by following [these intructions](../README.md#selfies-ted)
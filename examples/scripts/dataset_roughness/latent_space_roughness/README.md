# Latent space roughness analysis

In this example we demonstrate how TopSearch can be used to analyze the latent space of an autoencoder model, and generate new data to refine its learnt representation.  We apply this approach to analyze a smaller version of the SELFIES-TED model. 

The steps in this process are:

1. [Data generation](data_generation/README.md)
2. [Landscape exploration](landscape_exploration/README.md)
3. [Roughness surface samping](roughness_sampling/README.md)
4. [Fill-tuning](fill_tuning/README.md)

As many of these steps require significant compute time, expected outputs are supplied in each directory. You can run subsequent steps using these outputs.

## SELFIES-TED
In order to run the data generation or fill-tuning steps, you must download the SELFIES-TED code as follows:

1. Clone the [IBM foundation models for materials repo](https://github.com/IBM/materials)
2. Install the dependencies from the [SELFIES-TED requirements.txt](https://github.com/IBM/materials/blob/main/models/selfies_ted/requirements.txt) file into the same virtual environment as topsearch. 
```pip install -r materials/models/selfies-ted/requirements.txt```
3. Add the materails repo to your PYTHONPATH
```export PYTHONPATH=<IBM materials repo path>:$PYTHONPATH



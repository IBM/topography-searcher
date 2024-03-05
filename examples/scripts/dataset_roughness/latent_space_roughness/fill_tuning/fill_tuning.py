from topsearch.generation.generate_data import DataGenerator
from topsearch.data.kinetic_transition_network import KineticTransitionNetwork
from topsearch.analysis.minima_properties import get_ordered_minima
from models.selfies_ted.train import train
import numpy as np
import os


ktn = KineticTransitionNetwork(dump_path="../roughness_sampling/expected_output", dump_suffix=".roughness")
ktn.read_network()

# Scale minima to full range of values in the embedding space.
# Ignore any with small energies.

new_embeddings = []
threshold = -1e-1
for i in get_ordered_minima(ktn):
    if ktn.get_minimum_energy(i) < threshold:
        new_embeddings.append(ktn.get_minimum_coords(i) * 2 -1)
  
# Decode the minima to generate SELFIES for fill-tuning
generator = DataGenerator(128, -1, 1)
new_selfies = generator.decode_bart(new_embeddings, return_selfies=True)
deduped_selfies = list(dict.fromkeys(new_selfies))
selfies = [selfie.replace('][', '] [') for selfie in deduped_selfies]

np.savetxt("top_100_new_selfies.csv", selfies[0:100], fmt="%s", header="SELFIES", comments='')



train("top_100_new_selfies.csv", "ibm/materials.selfies-ted2m", 1)




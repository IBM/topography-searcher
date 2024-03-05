from topsearch.generation.generate_data import DataGenerator
import numpy as np

data_generator = DataGenerator(128, -1, 1)

training = data_generator.latin_hypercube_sample(5000)
response = data_generator.get_neighbour_similarity_dataset(training)

np.savetxt("./selfies-ted-mini-training.txt", training)
np.savetxt("./selfies-ted-mini-response.txt", response)


from typing import Callable, List, Optional
import numpy as np
from scipy.stats.qmc import LatinHypercube, scale
from scipy.stats import uniform_direction
from rdkit.Chem import AllChem, DataStructs
from transformers.modeling_outputs import BaseModelOutput
from models.selfies_ted.embedding import SELFIESForEmbeddingExploration

import torch
import selfies

import os


class DataGenerator():
    def __init__(self, dimensions: int, lower_bound: float = 0, upper_bound: float = 1, model: SELFIESForEmbeddingExploration = None) -> None:
        self.dimensions = dimensions
        self.sampler = LatinHypercube(self.dimensions)
        self.neighbour_sampler = uniform_direction(self.dimensions)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        if model is None:
            self.model = SELFIESForEmbeddingExploration.from_pretrained("ibm/materials.selfies-ted2m")
        else:
            self.model = model    

    def latin_hypercube_sample(self, samples: int) -> np.ndarray:
       embeddings = self.sampler.random(samples)
       return scale(embeddings, self.lower_bound, self.upper_bound)
   
    def create_neighbours(self, target_embedding: np.ndarray, neighbours: int, neighbour_distance=0.05) -> np.ndarray:
        # Sample neighbours from the surfacd of hypersphere of radisu neighbour_distance
        samples = self.neighbour_sampler.rvs(neighbours)
        neighbours = np.clip(target_embedding + samples * neighbour_distance, self.lower_bound, self.upper_bound)
        
        return neighbours
   
    def get_mean_neighbour_similarity(self, target_embedding: Optional[np.ndarray] = None, neighbours: Optional[np.ndarray] = None, target_smile: Optional[str] = None, neighbour_smiles: Optional[List[str]] = None, decoder: Optional[Callable[[np.ndarray], List[str]]] = None):
        if decoder is None:
             decoder = self.decode_bart
        if target_smile is None:
            target_smile = decoder([target_embedding])[0]
        if neighbour_smiles is None:
            neighbour_smiles = decoder(neighbours)

        target_mol = AllChem.MolFromSmiles(target_smile)
        if target_mol is None:
            return 0
        target_fp = AllChem.GetMorganFingerprintAsBitVect(target_mol, 3)
        neighbour_similarity = 0
        for n in neighbour_smiles:
            neighbour_mol = AllChem.MolFromSmiles(n)
            if neighbour_mol is None:
                continue
            neighbour_fp =  AllChem.GetMorganFingerprintAsBitVect(neighbour_mol, 3)
            similarity = DataStructs.FingerprintSimilarity(target_fp, neighbour_fp)
            neighbour_similarity += similarity

        return neighbour_similarity/len(neighbour_smiles)   
    
    def decode_bart(self, embedding: np.ndarray, return_selfies=False):
        # Project embedding vector to BART latent space (sequence length x hidden_size) 
        decoded_selfies = self.model.decode(torch.tensor(embedding, dtype=torch.float32))
        if return_selfies:
            return decoded_selfies
        return [selfies.decoder(decoded_selfie.replace("] [", "][")) for decoded_selfie in decoded_selfies]
    
    def get_neighbour_similarity_dataset(self, embeddings: np.ndarray, neighbour_distance=0.05, results_as_dict=False, decoder: Optional[Callable[[np.ndarray], List[str]]] = None, target_smiles: List[str]=None) -> np.ndarray | dict:
        if decoder is None:
            decoder = self.decode_bart

        NUM_NEIGHBOURS = 10
        similarities = np.zeros(len(embeddings))
        neighbour_embeddings = np.zeros((len(embeddings), NUM_NEIGHBOURS, self.dimensions))
        print("Generating neighbour embeddings...")
        for (idx, target_embedding) in enumerate(embeddings):
            neighbour_embeddings[idx] = self.create_neighbours(target_embedding, 10, neighbour_distance)

        print("Decoding neighbour embeddings...")
        batch_size = 1024
        all_neighbour_smiles = []
        for i in range(0, len(embeddings), batch_size):
            all_neighbour_smiles.extend(decoder(neighbour_embeddings[i:i+batch_size].reshape((-1, self.dimensions))))
            
        neighbour_smiles = np.array(all_neighbour_smiles).reshape(len(embeddings), NUM_NEIGHBOURS)

        print("Calculating neighbour similarity...")
        for idx, (target_embedding, neighbours_for_embedding) in enumerate(zip(embeddings, neighbour_smiles)):
            if target_smiles:
                similarities[idx] = self.get_mean_neighbour_similarity(target_embedding=target_embedding, neighbour_smiles=neighbours_for_embedding, target_smile=target_smiles[idx])
            else:
                similarities[idx] = self.get_mean_neighbour_similarity(target_embedding=target_embedding, neighbour_smiles=neighbours_for_embedding, decoder=decoder)

        if results_as_dict:
            return {"similarities": similarities}
        else:
            return similarities
from assertpy import assert_that
import numpy as np
from numpy.testing import assert_array_equal
import pytest

from topsearch.generation.generate_data import DataGenerator

@pytest.fixture
def data_generator() -> DataGenerator:
    return DataGenerator(128, -1, 1)

def test_latin_hypercube_sample_single_datapoint_dimensions(data_generator):
    data = data_generator.latin_hypercube_sample(1)
    assert_that(data.shape).is_equal_to((1,128))

def test_latin_hypercube_sample_multiple_datapoints_dimensions(data_generator):
    data = data_generator.latin_hypercube_sample(5)
    assert_that(data.shape).is_equal_to((5,128))  

def test_latin_hypercube_sample_bounds(data_generator):
    data = data_generator.latin_hypercube_sample(5)
    assert_that(np.all(data >= -1)).is_true()
    assert_that(np.all(data <= 1)).is_true()

def test_create_neighbours_returns_specified_number_of_neighbours(data_generator):
    embedding = np.full([128], 0.5)
    assert_that(data_generator.create_neighbours(embedding, 1)).is_length(1)

def test_create_neighbours_have_specified_dimensions():
    data_generator = DataGenerator(dimensions=16)
    embedding = np.full([16], 0.5)
    assert_that(data_generator.create_neighbours(embedding, 1).shape).is_equal_to((1,16))

def test_neighbours_are_within_specified_distance(data_generator):
    embedding = np.full([128], 0.5)
    neighbours = data_generator.create_neighbours(embedding, 10, 0.1)
    for neighbour in neighbours:
        for dimension in neighbour:
            assert_that(dimension).is_less_than(0.6)
            assert_that(dimension).is_greater_than(0.4)

def test_neighbours_are_within_default_bounds(data_generator):
    embedding = np.full([128], -1)
    neighbours = data_generator.create_neighbours(embedding, 10, 0.1)
    for neighbour in neighbours:
        for dimension in neighbour:
            assert_that(dimension).is_greater_than_or_equal_to(-1).is_less_than_or_equal_to(1)

def test_neighbours_are_within_lower_bounds_specified():
    data_generator = DataGenerator(dimensions=16, lower_bound=-0.5, upper_bound=3)
    embedding = np.full([16], -0.5)
    neighbours = data_generator.create_neighbours(embedding, 100, 0.1)
    assert_that(np.min(neighbours)).is_greater_than_or_equal_to(-0.5).is_less_than(0)
                
def test_neighbours_are_within_upper_bounds_specified():
    data_generator = DataGenerator(dimensions=16, lower_bound=-0.5, upper_bound=3)
    embedding = np.full([16], 3)
    neighbours = data_generator.create_neighbours(embedding, 100, 0.1)
    assert_that(np.max(neighbours)).is_less_than_or_equal_to(3).is_greater_than(2)

def test_neighbour_similarity_is_one_for_identical_embeddings(data_generator):
    target_embedding = np.full(128,0.5)
    neighbours = np.full([10,128], 0.5)

    assert_that(data_generator.get_mean_neighbour_similarity(target_embedding, neighbours)).is_equal_to(1)

def test_neighbour_similarity_is_low_for_opposite_embeddings(data_generator):
    target_embedding = np.full(128,0.5)
    neighbours = np.full([10,128], -0.5)

    assert_that(data_generator.get_mean_neighbour_similarity(target_embedding, neighbours)).is_less_than(0.1)    

def test_neighbour_similarity_is_medium_for_mixture_of_embeddings(data_generator):
    target_embedding = np.full([128],0, dtype=np.float32), 
    same_neighbours = np.full([5,128], 0, dtype=np.float32)
    far_neighbours = np.full([5,128], 0.5, dtype=np.float32)
    neighbours = np.concatenate((same_neighbours, far_neighbours))

    assert_that(data_generator.get_mean_neighbour_similarity(target_embedding, neighbours)).is_less_than(0.7).is_greater_than(0.3)   

def test_neighbour_similarity_with_matching_smiles_gives_1(data_generator):
    target_smile = "[C]"
    neighbour_smiles = [target_smile for _ in range(10)]

    assert_that(data_generator.get_mean_neighbour_similarity(target_smile=target_smile, neighbour_smiles=neighbour_smiles)).is_equal_to(1)

def test_neighbour_similarity_calls_specified_decoder_function(data_generator, mocker):
    target_embedding = np.full(128,0.5)
    neighbours = np.full([10,128], 0.5)

    decoder = mocker.MagicMock(return_value="C")
    assert_that(data_generator.get_mean_neighbour_similarity(target_embedding=target_embedding, neighbours=neighbours, decoder=decoder)).is_equal_to(1)
    assert_array_equal(decoder.call_args_list[0].args[0], [target_embedding])
    assert_array_equal(decoder.call_args_list[1].args[0], neighbours)

@pytest.fixture()
def mock_create_neighbours(data_generator, mocker):
    neighbours = np.full([10,128], 0.5)
    return mocker.patch.object(data_generator, 'create_neighbours', return_value=neighbours)

@pytest.fixture
def mock_get_mean_neighbour_similarity(data_generator, mocker):
    return mocker.patch.object(data_generator,'get_mean_neighbour_similarity', return_value=12)

def test_similarity_dataset_gives_one_output_for_single_embeddings(mock_create_neighbours, data_generator):
    embeddings = np.full([1,128], 0.5)

    assert_that(np.shape(data_generator.get_neighbour_similarity_dataset(embeddings))).is_equal_to((1,))

def test_similarity_dataset_gives_expected_similarity_single_datapoint(mock_create_neighbours, mock_get_mean_neighbour_similarity, data_generator):
    embedding = np.full([1,128], 0.5)

    assert_that(data_generator.get_neighbour_similarity_dataset(embedding)).is_equal_to(12)

def test_similarity_dataset_gets_neighbours_single_datapoint(mock_create_neighbours, data_generator):
    embedding = np.full([1,128], 0.5)

    data_generator.get_neighbour_similarity_dataset(embedding)

    assert_array_equal(mock_create_neighbours.call_args[0][0], embedding[0])

def test_similarity_dataset_gives_expected_similarity_multiple_datapoints(mocker, mock_create_neighbours, data_generator):
    embeddings = np.full([2,128], 0.5)
    expected_similarity = np.array([0, 12])
    mocker.patch.object(data_generator, 'get_mean_neighbour_similarity', side_effect=expected_similarity)
    
    assert_array_equal(data_generator.get_neighbour_similarity_dataset(embeddings), expected_similarity)

def test_similarity_dataset_gets_neighbours_multiple_datapoints(mock_create_neighbours, data_generator):
    embedding1 = np.full([1,128], 0.5)
    embedding2 = np.full([1,128], 0)

    data_generator.get_neighbour_similarity_dataset(np.concatenate((embedding1, embedding2)))

    assert_array_equal(mock_create_neighbours.call_args_list[0].args[0], embedding1[0])
    assert_array_equal(mock_create_neighbours.call_args_list[1].args[0], embedding2[0])

def test_similarity_dataset_gives_expected_similarity_with_specified_decoder(mocker, data_generator):
    embeddings = np.full([2,128], 0.5)
    expected_similarity = np.array([1, 1])
    decoder = mocker.MagicMock(side_effect=[["C"]*20, ["C"], ["C"]])
    
    assert_array_equal(data_generator.get_neighbour_similarity_dataset(embeddings, decoder=decoder), expected_similarity)
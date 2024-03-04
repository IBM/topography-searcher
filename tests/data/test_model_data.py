import pytest
import numpy as np
import os.path
from topsearch.data.model_data import ModelData

###### SMALL DATASET ##########

def test_initialisation():
    model_data = ModelData(training_file='test_data/training_model1.txt',
                           response_file='test_data/response_model1.txt')
    assert model_data.n_points == 5
    assert model_data.training.shape == (5, 2)
    assert model_data.response.shape == (5, )

def test_write_data():
    model_data = ModelData(training_file='test_data/training_model1.txt',
                           response_file='test_data/response_model1.txt')
    model_data.write_data('training_out.txt', 'response_out.txt')
    model_data2 = ModelData(training_file='training_out.txt',
                            response_file='response_out.txt')
    assert np.all(model_data.training == model_data2.training)
    assert np.all(model_data.response == model_data2.response)
    os.remove('training_out.txt')
    os.remove('response_out.txt')

def test_update_data():
    model_data = ModelData(training_file='test_data/training_model1.txt',
                           response_file='test_data/response_model1.txt')
    extra_training = np.array([[0.2, 1.0]])
    extra_response = np.array([7.4])
    model_data.append_data(extra_training, extra_response)
    assert model_data.n_points == 6
    assert np.all(model_data.training[5, :] == [0.2, 1.0])
    assert model_data.response[5] == 7.4

def test_update_data2():
    model_data = ModelData(training_file='test_data/training_model1.txt',
                           response_file='test_data/response_model1.txt')
    extra_training = np.array([[0.2, 1.0], [0.3, 1.0], [4.2, -0.3]])
    extra_response = np.array([0.1, 0.2, 0.3])
    model_data.append_data(extra_training, extra_response)
    assert model_data.n_points == 8
    assert np.all(model_data.training[6, :] == [0.3, 1.0])
    assert model_data.response[6] == 0.2

def test_limit_response_maximum():
    model_data = ModelData(training_file='test_data/training_model1.txt',
                           response_file='test_data/response_model1.txt')
    model_data.limit_response_maximum(7.5)
    assert np.all(model_data.response == np.array([7.5, 3.1, 7.5, -0.3, 0.0]))

def test_standardise_response():
    model_data = ModelData(training_file='test_data/training_model1.txt',
                           response_file='test_data/response_model1.txt')
    model_data.standardise_response()
    assert np.std(model_data.response) == pytest.approx(1.0)
    assert np.mean(model_data.response) == pytest.approx(0.0)

def test_standardise_training():
    model_data = ModelData(training_file='test_data/training_model1.txt',
                           response_file='test_data/response_model1.txt')
    model_data.standardise_training()
    assert np.all(np.std(model_data.training, axis=0) == 
                  pytest.approx(np.array([1.0, 1.0])))
    assert np.all(np.mean(model_data.training, axis=0) == 
                  pytest.approx(np.array([0.0, 0.0])))

def test_normalise_response():
    model_data = ModelData(training_file='test_data/training_model1.txt',
                           response_file='test_data/response_model1.txt')
    model_data.normalise_response()
    assert np.max(model_data.response) == pytest.approx(1.0)
    assert np.min(model_data.response) == pytest.approx(0.0)

def test_normalise_training():
    model_data = ModelData(training_file='test_data/training_model1.txt',
                           response_file='test_data/response_model1.txt')
    model_data.normalise_training()
    assert np.all(np.max(model_data.training, axis=0) == 
                  pytest.approx(np.array([1.0, 1.0])))
    assert np.all(np.min(model_data.training, axis=0) == 
                  pytest.approx(np.array([0.0, 0.0])))

def test_remove_duplicates():
    model_data = ModelData(training_file='test_data/training_model1.txt',
                           response_file='test_data/response_model1.txt')
    model_data.remove_duplicates(dist_cutoff = 1e-1)
    assert model_data.n_points == 4
    assert np.all(model_data.training ==
                  np.array([[-0.5, 1.2], [1.0, 1.4], [2.5, 1.6], [-0.3, 0.2]]))
    assert np.all(model_data.response ==
                  np.array([8.0, 3.1, 10.2, 0.0]))

def test_unnormalise_response():
    model_data = ModelData(training_file='test_data/training_model1.txt',
                           response_file='test_data/response_model1.txt')
    model_data.normalise_response()
    model_data.unnormalise_response()
    model_data2 = ModelData(training_file='test_data/training_model1.txt',
                           response_file='test_data/response_model1.txt')
    assert np.all(model_data.training == pytest.approx(model_data2.training))
    assert np.all(model_data.response == pytest.approx(model_data2.response))

def test_unnormalise_training():
    model_data = ModelData(training_file='test_data/training_model1.txt',
                           response_file='test_data/response_model1.txt')
    model_data.normalise_training()
    model_data.unnormalise_training()
    model_data2 = ModelData(training_file='test_data/training_model1.txt',
                           response_file='test_data/response_model1.txt')
    assert np.all(model_data.training == pytest.approx(model_data2.training))
    assert np.all(model_data.response == pytest.approx(model_data2.response))

def test_unstandardise_response():
    model_data = ModelData(training_file='test_data/training_model1.txt',
                           response_file='test_data/response_model1.txt')
    model_data.standardise_response()
    model_data.unstandardise_response()
    model_data2 = ModelData(training_file='test_data/training_model1.txt',
                           response_file='test_data/response_model1.txt')
    assert np.all(model_data.training == pytest.approx(model_data2.training))
    assert np.all(model_data.response == pytest.approx(model_data2.response))

def test_unstandardise_training():
    model_data = ModelData(training_file='test_data/training_model1.txt',
                           response_file='test_data/response_model1.txt')
    model_data.standardise_training()
    model_data.unstandardise_training()
    model_data2 = ModelData(training_file='test_data/training_model1.txt',
                           response_file='test_data/response_model1.txt')
    assert np.all(model_data.training == pytest.approx(model_data2.training))
    assert np.all(model_data.response == pytest.approx(model_data2.response))

def test_point_in_hull():
    model_data = ModelData(training_file='test_data/training_model1.txt',
                           response_file='test_data/response_model1.txt')
    model_data.convex_hull()
    point1 = np.array([1.0, 1.0])
    in_hull = model_data.point_in_hull(point1)
    assert in_hull == True
    point2 = np.array([0.0, 0.5])
    in_hull = model_data.point_in_hull(point2)
    assert in_hull == True
    point3 = np.array([1.0, 0.0])
    in_hull = model_data.point_in_hull(point3)
    assert in_hull == False

###### LARGER DATASET ###########

def test_initialisation2():
    model_data = ModelData(training_file='test_data/training_model2.txt',
                           response_file='test_data/response_model2.txt')
    assert model_data.n_points == 150
    assert model_data.training.shape == (150, 3)
    assert model_data.response.shape == (150, )

def test_write_data2():
    model_data = ModelData(training_file='test_data/training_model2.txt',
                           response_file='test_data/response_model2.txt')
    model_data.write_data('training2_out.txt', 'response2_out.txt')
    model_data2 = ModelData(training_file='training2_out.txt',
                            response_file='response2_out.txt')
    assert np.all(model_data.training == model_data2.training)
    assert np.all(model_data.response == model_data2.response)
    os.remove('training2_out.txt')
    os.remove('response2_out.txt')

def test_update_data3():
    model_data = ModelData(training_file='test_data/training_model2.txt',
                           response_file='test_data/response_model2.txt')
    extra_training = np.array([[0.2, 1.0, 0.8]])
    extra_response = np.array([7.4])
    model_data.append_data(extra_training, extra_response)
    assert model_data.n_points == 151
    assert np.all(model_data.training[150, :] == [0.2, 1.0, 0.8])
    assert model_data.response[150] == 7.4

def test_update_data4():
    model_data = ModelData(training_file='test_data/training_model2.txt',
                           response_file='test_data/response_model2.txt')
    extra_training = np.array([[0.2, 1.0, 0.4], [0.3, 1.0, 1.2],
                               [4.2, -0.3, -0.4]])
    extra_response = np.array([0.1, 0.2, 0.3])
    model_data.append_data(extra_training, extra_response)
    assert model_data.n_points == 153
    assert np.all(model_data.training[152, :] == [4.2, -0.3, -0.4])
    assert model_data.response[152] == 0.3

def test_limit_response_maximum2():
    model_data = ModelData(training_file='test_data/training_model2.txt',
                           response_file='test_data/response_model2.txt')
    model_data.limit_response_maximum(0.2)
    assert np.max(model_data.response) == 0.2

def test_standardise_response2():
    model_data = ModelData(training_file='test_data/training_model2.txt',
                           response_file='test_data/response_model2.txt')
    model_data.standardise_response()
    assert np.std(model_data.response) == pytest.approx(1.0)
    assert np.mean(model_data.response) == pytest.approx(0.0)

def test_standardise_training2():
    model_data = ModelData(training_file='test_data/training_model2.txt',
                           response_file='test_data/response_model2.txt')
    model_data.standardise_training()
    assert np.all(np.std(model_data.training, axis=0) == 
                  pytest.approx(np.array([1.0, 1.0, 1.0])))
    assert np.all(np.mean(model_data.training, axis=0) == 
                  pytest.approx(np.array([0.0, 0.0, 0.0])))

def test_normalise_response2():
    model_data = ModelData(training_file='test_data/training_model2.txt',
                           response_file='test_data/response_model2.txt')
    model_data.normalise_response()
    assert np.max(model_data.response) == pytest.approx(1.0)
    assert np.min(model_data.response) == pytest.approx(0.0)

def test_normalise_training2():
    model_data = ModelData(training_file='test_data/training_model2.txt',
                           response_file='test_data/response_model2.txt')
    model_data.normalise_training()
    assert np.all(np.max(model_data.training, axis=0) == 
                  pytest.approx(np.array([1.0, 1.0, 1.0])))
    assert np.all(np.min(model_data.training, axis=0) == 
                  pytest.approx(np.array([0.0, 0.0, 0.0])))

def test_remove_duplicates2():
    model_data = ModelData(training_file='test_data/training_model2.txt',
                           response_file='test_data/response_model2.txt')
    model_data.remove_duplicates(dist_cutoff = 1e-1)
    assert model_data.n_points == 138

def test_unnormalise_response2():
    model_data = ModelData(training_file='test_data/training_model2.txt',
                           response_file='test_data/response_model2.txt')
    model_data.normalise_response()
    model_data.unnormalise_response()
    model_data2 = ModelData(training_file='test_data/training_model2.txt',
                            response_file='test_data/response_model2.txt')
    assert np.all(model_data.training == pytest.approx(model_data2.training))
    assert np.all(model_data.response == pytest.approx(model_data2.response))

def test_unnormalise_training2():
    model_data = ModelData(training_file='test_data/training_model2.txt',
                           response_file='test_data/response_model2.txt')
    model_data.normalise_training()
    model_data.unnormalise_training()
    model_data2 = ModelData(training_file='test_data/training_model2.txt',
                            response_file='test_data/response_model2.txt')
    assert np.all(model_data.training == pytest.approx(model_data2.training))
    assert np.all(model_data.response == pytest.approx(model_data2.response))

def test_unstandardise_response2():
    model_data = ModelData(training_file='test_data/training_model2.txt',
                           response_file='test_data/response_model2.txt')
    model_data.standardise_response()
    model_data.unstandardise_response()
    model_data2 = ModelData(training_file='test_data/training_model2.txt',
                            response_file='test_data/response_model2.txt')
    assert np.all(model_data.training == pytest.approx(model_data2.training))
    assert np.all(model_data.response == pytest.approx(model_data2.response))

def test_unstandardise_training2():
    model_data = ModelData(training_file='test_data/training_model2.txt',
                           response_file='test_data/response_model2.txt')
    model_data.standardise_training()
    model_data.unstandardise_training()
    model_data2 = ModelData(training_file='test_data/training_model2.txt',
                            response_file='test_data/response_model2.txt')
    assert np.all(model_data.training == pytest.approx(model_data2.training))
    assert np.all(model_data.response == pytest.approx(model_data2.response))

def test_feature_subset():
    model_data = ModelData(training_file='test_data/training_model3.txt',
                           response_file='test_data/response_model3.txt')
    features = [0, 2, 4]
    model_data.feature_subset(features)
    assert np.all(model_data.training == pytest.approx(np.array([[0.89332, 1.0,  2.0 ],
                                                                 [1.8785,  2.0,  0.0 ],
                                                                 [3.21484, 2.0,  6.0 ],
                                                                 [2.5657,  1.0,  5.0 ],
                                                                 [4.04402, 2.0,  6.0 ]])))
    assert model_data.n_dims == 3

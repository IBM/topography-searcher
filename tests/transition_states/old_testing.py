def uphill_step_taking():
    """ Check that the step size is calculated appropriately and the
        correct step is taken to a new position """

    # Initialise class instances
    camelback = Camelback(ndim=2, bounds=[(-3.0,3.0),(-2.0,2.0)])
    minimiser = LBFGS(camelback, 1e-6, 5, 500)
    single_ended_search = HybridEigenvectorFollowing(camelback, minimiser,
                                                     1e-4, 250, 1e-2)

    # Take non-stationary point and locate lowest eigenvector and eigenvalue
    coords = np.array([0.8, 0.1])
    eigenvalue, eigenvector = camelback.hessian_lowest_eigenvector(coords)
    # Take uphill step in that direction
    updated_coords = single_ended_search.take_uphill_step(coords, eigenvector,
                                                          eigenvalue)

    # Check appropriate step length has been chosen
    assert updated_coords[0] - 0.8031842668579567 < 1e-5
    assert updated_coords[1] - 0.11195487514819265 < 1e-5

    # Test second minimum
    coords = np.array([0.3, 0.3])
    eigenvalue, eigenvector = camelback.hessian_lowest_eigenvector(coords)
    # Take uphill step in that direction
    updated_coords = single_ended_search.take_uphill_step(coords, eigenvector,
                                                          eigenvalue)

    # Check appropriate step length has been chosen
    assert updated_coords[0] - 0.3103636018006611 < 1e-5
    assert updated_coords[1] - 0.20053847096632116 < 1e-5

    coords = np.array([0.3, 0.3])
    # Repeat for the negative of the eigenvector
    eigenvector = -1.0*eigenvector

    # Take uphill step in that direction
    updated_coords = single_ended_search.take_uphill_step(coords, eigenvector,
                                                          eigenvalue)

    # Check appropriate step length has been chosen
    assert updated_coords[0] - 0.3103636018006611 < 1e-5
    assert updated_coords[1] - 0.20053847096632116 < 1e-5

def test_update_eigenvector_bounds():
    """ Check if point at bounds and update the corresponding 
        eigenvector bounds """
    
    # Initialise class instances
    camelback = Camelback(ndim=2, bounds=[(-3.0,3.0),(-2.0,2.0)])
    minimiser = LBFGS(camelback, 1e-6, 5, 500)
    single_ended_search = HybridEigenvectorFollowing(camelback, minimiser,
                                                     1e-4, 250, 1e-2)

    # Select a point at both bounds
    coords = np.array([3.0, 2.0])
    # Test for if point is at any bounds
    at_bounds, active_bounds = \
        single_ended_search.check_active_bounds(coords)
    # Update eigenvector bounds
    single_ended_search.update_eigenvector_bounds(active_bounds)

    # Check we recognise the bounds and eigenvector bounds are appropriate
    assert at_bounds == True
    assert single_ended_search.eig_bounds == [(np.NINF, 0.0), (np.NINF, 0.0)]

def test_find_pushoff():
    """ Check we find appropriate pushoff size from transition state """

    # Initialise class instances
    camelback = Camelback(ndim=2, bounds=[(-3.0,3.0),(-2.0,2.0)])
    minimiser = LBFGS(camelback, 1e-6, 5, 500)
    single_ended_search = HybridEigenvectorFollowing(camelback, minimiser,
                                                     1e-4, 250, 1e-2)

    # Specify valid transition state and downhill eigenvector
    transition_state = np.array([1.109205312695322965, -0.7682680828827476160])
    eigenvector = np.array([-0.99937336, 0.03539617])

    # Find the pushoff needed to give lower energies
    positive_x, negative_x = \
        single_ended_search.find_pushoff(transition_state, eigenvector)

    # Check that single push was necessary and gave coordinates
    assert positive_x == pytest.approx(np.array([1.09921158, -0.76791412]))
    assert negative_x == pytest.approx(np.array([1.11919905, -0.76862204]))

def test_do_pushoff():
    """ Check that the pushoff is followed correctly """

    # Initialise class instances
    camelback = Camelback(ndim=2, bounds=[(-3.0,3.0),(-2.0,2.0)])
    minimiser = LBFGS(camelback, 1e-6, 5, 500)
    single_ended_search = HybridEigenvectorFollowing(camelback, minimiser,
                                                     1e-4, 250, 1e-2)

    # Specify parameters for pushing off from transition state
    transition_state = np.array([0.0, 0.0])
    eigenvector = np.array([0.0, 1.0])
    increment = 5e-3
    iteration = 2
    # Push along the downhill eigenvector by increment after initial
    # pushoff
    moved_x = single_ended_search.do_pushoff(transition_state,
                                             eigenvector,
                                             increment, iteration)

    # Check expected coordinates after displacement
    assert moved_x == pytest.approx(np.array([0.0, 0.02]))

    # Attempt again for different eigenvector
    eigenvector = np.array([0.5, 0.5])
    increment = 5e-3
    iteration = 3
    moved_x = single_ended_search.do_pushoff(transition_state,
                                             eigenvector,
                                             increment, iteration)

    # Check expected coordinates after displacement    
    assert moved_x == pytest.approx(np.array([0.0125, 0.0125]))

def test_steepest_descent():
    """ Test the finding of pushoffs and the subsequent local minimisations
        to find connected minima """

    # Initialise class instances
    camelback = Camelback(ndim=2, bounds=[(-3.0,3.0),(-2.0,2.0)])
    minimiser = LBFGS(camelback, 1e-6, 5, 500)
    single_ended_search = HybridEigenvectorFollowing(camelback, minimiser,
                                                     1e-4, 250, 1e-2)

    # Set valid transition state and downhill eigenvector
    transition_state = np.array([1.109205312695322965, -0.7682680828827476160])
    eigenvector = np.array([-0.99937336, 0.03539617])

    # Perform steepest descent paths after finding pushoff
    plus_min, plus_energy, minus_min, minus_energy = \
        single_ended_search.steepest_descent(transition_state, eigenvector)

    # Make sure steepest-descent paths locate the expected minima
    assert plus_min == pytest.approx(np.array([0.08984201, -0.71265638]))
    assert plus_energy == pytest.approx(-1.0316284534898736)
    assert minus_min == pytest.approx(np.array([1.70360671, -0.79608357]))
    assert minus_energy == pytest.approx(-0.21546382438371858)

def test_projection():
    """ Check the projection of an eigenvector onto the bounds """
    # Initialise class instances
    camelback = Camelback(ndim=2, bounds=[(-3.0,3.0),(-2.0,2.0)])
    minimiser = LBFGS(camelback, 1e-6, 5, 500)
    single_ended_search = HybridEigenvectorFollowing(camelback, minimiser,
                                                     1e-4, 250, 1e-2)
    
    eigenvector = np.array([-0.3, 0.1])
    active_bounds = np.array([-1, 0])
    proj = single_ended_search.projection_onto_bounds(eigenvector, active_bounds)
    
    assert np.array_equal(proj, np.array([0.0,1.0]))

    schwefel = Schwefel(ndim=3, bounds=[(-5.0,5.0),(-5.0,5.0),(-5.0,5.0)])
    minimiser = LBFGS(schwefel, 1e-6, 5, 500)
    single_ended_search = HybridEigenvectorFollowing(schwefel, minimiser,
                                                     1e-4, 250, 1e-2)
    
    eigenvector = np.array([0.3, -0.1, 0.2])
    active_bounds = np.array([1, 1, 0])
    proj = single_ended_search.projection_onto_bounds(eigenvector, active_bounds)

    assert np.abs(proj[0]) < 1e-5
    assert np.abs(proj[1] - -0.4472136) < 1e-5
    assert np.abs(proj[2] - 0.89442719) < 1e-5   

def test_ts_search_camel():
    """ Test the complete transition state search from a given point 
        for the two-dimensional six-hump camel function """

    # Initialise class instances
    camelback = Camelback(ndim=2, bounds=[(-3.0,3.0),(-2.0,2.0)])
    minimiser = LBFGS(camelback, 1e-6, 5, 500)
    single_ended_search = HybridEigenvectorFollowing(camelback, minimiser,
                                                     1e-4, 250, 1e-2)

    # Start from coordinates near to transition state (0.0, 0.0)
    coords = np.array([0.05, 0.1])
    # Perform transition state search
    coords, e_ts, plus_min, e_plus, minus_min, e_minus, eigenvector = \
        single_ended_search.single_ended_ts_search(coords)

    # Plus and minus are interchangeable so enforce ordering for comparison
    if plus_min[0] < 0.0:
        tmp_min = plus_min
        tmp_energy = e_plus
        plus_min = minus_min
        e_plus = e_minus
        minus_min = tmp_min
        e_minus = tmp_energy

    # Check transition state is at (0.0, 0.0) with zero energy
    assert np.abs(coords[0]) < 1e-2
    assert np.abs(coords[1]) < 1e-2
    assert np.abs(e_ts) < 1e-3
    # Check correct connected minima are identified
    assert plus_min == pytest.approx(np.array([0.08984201, -0.7126564]))
    assert minus_min == pytest.approx(np.array([-0.08984201, 0.7126564]))

def test_ts_search_schwefel():
    """ Test the complete transition state search from a point far from
        a transition state on the three-dimensional Schwefel function """

    # Initialise class instances
    schwefel = Schwefel(ndim=3, bounds=[(-500.0, 500.0),
                                        (-500.0, 500.0),
                                        (-500.0, 500.0)])
    minimiser = LBFGS(schwefel, 1e-6, 5, 500)
    single_ended_search = \
        HybridEigenvectorFollowing(potential=schwefel,
                                   minimiser=minimiser,
                                   conv_crit=1e-4,
                                   ts_steps=50,
                                   pushoff=5e-1,
                                   max_uphill_step_size=1.0,
                                   positive_eigenvalue_step=2.0,
                                   min_uphill_step_size=1e-7)

    # Start from coordinates near a minimum (124.829356,
    #  203.81425265, -124.829356419)
    coords = np.array([123.0, 21.0, -125.0])
    # Perform transition state search
    coords, e_ts, plus_min, e_plus, minus_min, e_minus, eigenvector = \
        single_ended_search.single_ended_ts_search(coords)

    # Plus and minus are interchangeable so enforce ordering for comparison
    if plus_min[0] > 66.0:
        tmp_min = plus_min
        tmp_energy = e_plus
        plus_min = minus_min
        e_plus = e_minus
        minus_min = tmp_min
        e_minus = tmp_energy

    # Check transition state is at correct position
    assert np.abs(coords[0] - 124.82935672) < 1e-2
    assert np.abs(coords[1] - 5.23920087) < 1e-2
    assert np.abs(coords[2] - -124.82935509) < 1e-2
    assert np.abs(e_ts - 1253.0033983747162) < 1e-3

def test_ts_search_acquisition():
    """ Test the complete transition state search on an acquistion function
        surface, fitted to the scaled Schwefel function with 45 data points.
        Acquisition functions can be difficult for optimisation due to 
        numerical instability and very small curvature """

    # Initialise class instances
    function_bounds = [(-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)]
    hyperparameter_bounds = [(2.0, 2.0), (2.0, 2.0),
                             (2.0, 2.0), (0.1, 0.1)]
    # Log marginal likelihood for fitting Gaussian process
    lml = LogMarginalLikelihood(kernel_choice='RBF',
                                training=acquisition_data.schwefel_3d_training/100.0,
                                response=acquisition_data.schwefel_3d_response,
                                bounds=hyperparameter_bounds)
    acquisition = Acquisition(gaussian_process=lml,
                              bounds=function_bounds,
                              acquisition_choice='UCB',
                              zeta=0.2)
    minimiser = LBFGS(acquisition, 1e-6, 5, 500)
    single_ended_search = \
        HybridEigenvectorFollowing(potential=acquisition,
                                   minimiser=minimiser,
                                   conv_crit=1e-4,
                                   ts_steps=50,
                                   pushoff=1e-2,
                                   max_uphill_step_size=1.0,
                                   positive_eigenvalue_step=2.0,
                                   min_uphill_step_size=5e-8)

    # Start from random coordinates in space
    coords = np.array([1.23, 0.21, -1.25])
    # Perform transition state search
    coords, e_ts, plus_min, e_plus, minus_min, e_minus, eigenvector = \
        single_ended_search.single_ended_ts_search(coords)

    # Check transition state is at expected position
    assert np.abs(coords[0] - 1.30305281) < 1e-1
    assert np.abs(coords[1] - 0.79727297) < 1e-1
    assert np.abs(coords[2] - -0.8744177) < 1e-1
    assert np.abs(e_ts - -0.3376015) < 1e-2

    # Pick second random point from elsewhere in space
    coords = np.array([-3.21, 1.21, 1.25])
    # Perform transition state search
    coords, e_ts, plus_min, e_plus, minus_min, e_minus, eigenvector = \
        single_ended_search.single_ended_ts_search(coords)

    # Check transition state is at expected position
    assert np.abs(coords[0] - -4.2245036) < 1e-1
    assert np.abs(coords[1] - 3.56679288) < 1e-1
    assert np.abs(coords[2] - 1.16928197) < 1e-1
    assert np.abs(e_ts - -0.45269675) < 1e-2

def test_ts_search_acquisition_lj():
    """ Test the complete transition state search for the scaled
        Schwefel function in three dimensions. This example has very
        high curvature and is a challenge for hybrid-eigenvector following """

    # Initialise class instances
    function_bounds = [(0.5, 3.0), (0.5, 3.0), (0.5, 3.0),
                       (0.5, 3.0), (0.5, 3.0), (0.5, 3.0)]
    hyperparameter_bounds = [(1.125, 1.125), (1.125, 1.125), (1.125, 1.125),
                             (1.125, 1.125), (0.5, 0.5), (0.5, 0.5),
                             (0.065477, 0.065477)]
    # Log marginal likelihood for fitting Gaussian process
    lml = LogMarginalLikelihood(kernel_choice='RBF',
                                training=acquisition_data.lj_training,
                                response=acquisition_data.lj_response,
                                bounds=hyperparameter_bounds)
    acquisition = Acquisition(gaussian_process=lml,
                              bounds=function_bounds,
                              acquisition_choice='UCB',
                              zeta=0.2)
    minimiser = LBFGS(acquisition, 1e-6, 5, 500)
    single_ended_search = \
        HybridEigenvectorFollowing(potential=acquisition,
                                   minimiser=minimiser,
                                   conv_crit=1e-4,
                                   ts_steps=75,
                                   pushoff=1e-2,
                                   max_uphill_step_size=1.0,
                                   positive_eigenvalue_step=2.0,
                                   min_uphill_step_size=5e-8)

    # Start from coordinates near a minimum (1.248293565484656087e+00,
    #  2.038142526514610253e+00 -1.248293564192949789e+00)
    coords = np.array([1.23, 0.7, 1.23, 2.4, 1.75, 1.75])
    # Perform transition state search
    coords, e_ts, plus_min, e_plus, minus_min, e_minus, eigenvector = \
        single_ended_search.single_ended_ts_search(coords)

    # Check transition state is at (0.0, 0.0) with zero energy
    assert np.abs(coords[0] - 1.53670101) < 1e-1
    assert np.abs(coords[1] - 1.86153428) < 1e-1
    assert np.abs(coords[2] - 1.72436339) < 1e-1
    assert np.abs(coords[3] - 1.12353052) < 1e-1
    assert np.abs(coords[4] - 1.78829708) < 1e-1
    assert np.abs(coords[5] - 1.62428251) < 1e-1
    assert np.abs(e_ts - -1.05975449) < 1e-2

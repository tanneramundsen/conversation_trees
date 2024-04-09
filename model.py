# Import necessary libraries
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from treelib import Tree
from random import choices

def popularity(tree_vec, time):
    '''
    Returns the popularity of the parent of the new node at index 'time_index'

    For a tree like
    1--2
    |  |__3
    |  |__4
    |     |__6
    |__5 

    The parent vector representation at each time stamp would be
    t_0 = []
    t_1 = [1]
    t_2 = [1, 2]
    t_3 = [1, 2, 2]
    t_4 = [1, 2, 2, 1]
    t_5 = [1, 2, 2, 1, 4]

    The node popularity calculation starts at t=2 and would look like
    t_2 = [1, 2]
        d_{2,2} = 1 + 0
        # Since we always start with 1 and the parent node 2 has appeared 0
        # times in the parent vector at t_1
    t_3 = [1, 2, 2]
        d_{3,3} = 1 + 1
        # Since we always start with 1 and the parent node 2 appears 1 times 
        # in the parent vector at t_2 
    t_4 = [1, 2, 2, 1]
        d_{4,4} = 1 + 1
    t_5 = [1, 2, 2, 1, 4]
        d_{5,5} = 1 + 0
    '''
    parent = tree_vec[time-1]
    return 1 + sum(tree_vec[:(time-1)] == parent)

def popularity(tree_vec, time, k):
    '''
    Returns the popularity of node 'k' at time 'time'
    '''
    if time==k:
        return 0
    
    return 1 + sum(tree_vec[:(time-1)] == k)

def novelty(tree_vec,time):
    '''
    Returns the novely of the parent of the new node at index 'time'
    Equals the age of the node at time 'time'
    '''
    return (time - tree_vec[time-1]) + 1

def type_bias(parent_type_vec,time):
    '''
    Returns 1 if the parent of the new node at index 'time' is the of type
    either 'announcement' or 'question'
    '''
    return int(parent_type_vec[time-1] == 1)

def root_bias(tree_vec,time):
    '''
    Returns 1 if the parent of the new node at index 'time' is the root node
    and 0 otherwise
    '''
    return int(tree_vec[time - 1] == 1)

def normalizing_factor_model_1(time, parameters):
    '''
    This function returns the normalizing factor for a given tree at a given time
    This factor allows the attractiveness function to be a valid probability

    'time_index' is the current time
    'parameters' are [alpha,beta,tau] from the log likelihood equation
    '''
    alpha = parameters[0]
    beta = parameters[1]
    tau = parameters[2]

    if np.abs(tau - 1) < 0.0001:
        n = 2*alpha*(time-1) + beta + 1
    else:
        n = 2*alpha*(time-1) + beta + (tau*(tau**(time) - 1))/(tau - 1)

    return n

def normalizing_factor_model_2(time, parameters, type_vec):
    '''
    This function returns the normalizing factor for a given tree at a given time
    This factor allows the attractiveness function to be a valid probability

    'time_index' is the current time
    'parameters' are [alpha,beta,tau] from the log likelihood equation
    'type_vec' is a a boolean array where type_vec[i] = 0 if that node is not a positive bias type and 1 if it is (including the root)
    '''
    alpha = parameters[0]
    beta = parameters[1]
    tau = parameters[2]
    rho = parameters[3]

    if np.abs(tau - 1) < 0.0001:
        n = 2*alpha*(time-1) + beta + 1 + rho*np.sum(type_vec[:time])
    else:
        n = 2*alpha*(time-1) + beta + (tau*(tau**(time) - 1))/(tau - 1) + rho*np.sum(type_vec[:time])

    return n

def attractiveness_function_model_1(tree_vec,time,k,parameters):
    '''
        Returns the attractiveness value (un-normalized) for model_1 for node 'k' at time 'time'
    '''
    # Extract the weights from 'parameters'
    alpha = parameters[0]
    beta = parameters[1]
    tau = parameters[2]

    popularity_k = popularity(tree_vec,time,k)
    novelty_k = (time - k) + 1
    root_bias_k = int(k==1)

    return alpha*popularity_k + beta*root_bias_k + tau**novelty_k

def attractiveness_function_model_2(tree_vec,type_vec,time,k,parameters):
    '''
        Returns the attractiveness value (un-normalized) for model_2 for node 'k' at time 'time'
        'k' is such that the root node is k=1
    '''
    alpha = parameters[0]
    beta = parameters[1]
    tau = parameters[2]
    rho = parameters[3]

    popularity_k = popularity(tree_vec,time,k)
    novelty_k = (time - k) + 1
    root_bias_k = int(k==1)
    type_bias_k = int(type_vec[k-1]==1)

    return alpha*popularity_k + beta*root_bias_k + tau**novelty_k + rho*type_bias_k

def degree_probability_mass(tree_vec,time,parameters,type_vec=None):
    
    is_model_1 = type_vec==None

    pmf = np.zeros(time) # a 1-D array to hold the probability values for each node that exists at time t
    for k in range(1,time+1):
        if is_model_1:
            pmf[k-1] = attractiveness_function_model_1(tree_vec,time,k,parameters)
        else:
            pmf[k-1] = attractiveness_function_model_2(tree_vec,type_vec,time,k,parameters)
    
    if is_model_1:
        pmf /= normalizing_factor_model_1(time,parameters)
    else:
        pmf /= normalizing_factor_model_2(time,parameters,type_vec)
    
    return pmf

def build_synthetic_tree(length,parameters,type_vec=None):
    '''
        if 'type_vec' is not-specified, model_1, otherwise, model_2
        this method builds a tree of size 'length' according the model paramaterized by 'parameters'
    '''
    tree = Tree()
    if length == 0:
        return(tree,np.array([]))
    
    tree.create_node(1,1)
    synthetic_parent_vec = np.zeros(length-1)
    if length > 1:
        synthetic_parent_vec[0] = 1 # all trees start with 1 root node and a second node attached to the root 
        tree.create_node(2,2,1)
        for t in range(2,length):
            cur_pmf = degree_probability_mass(synthetic_parent_vec,t,parameters,type_vec)
            population = np.arange(1,t+1)
            selected_parent = choices(population,weights=cur_pmf,k=1)
            synthetic_parent_vec[t-1] = selected_parent[0]
            tree.create_node(t+1,t+1,selected_parent[0])

    return (tree,synthetic_parent_vec)

# negative log likelihood function for model_1
def negative_log_likelihood_model_1(parameters, data):
    '''
    'parameters' are the parameters of our likelihood (derived from our attractiveness function)
        'parameters' = [alpha, beta, tau]
    'data' is a list of trees in parent vector format meaning
    For a tree like
    1--2
    |  |__3
    |  |__4
    |     |__6
    |__5 

    tree_vec = [1, 2, 2, 1, 4]

    This calculates the negative log likelihood of the data given the parameters.
    The negative log likelihood is:
        l(data|parameters) = sum_{i=1}^{n} l(tree)
        Where
        l(tree) = sum_{t=2}^{|tree|} phi(tree,t) - log(Z_t)
        Where
        1) phi is the Attractiveness function
            phi(tree,t) = alpha*popularity + beta*root_bias + tau**(novelty)
        3) Z_t is the normalizing factor at time t
    '''
    
    # Initialize negative log likelihood
    log_likelihood = 0.0

    # Extract the weights from 'parameters'
    alpha = parameters[0]
    beta = parameters[1]
    tau = parameters[2]

    # Loop over each tree in the dataset
    for tree in data:        
        # Calculate the negative log likelihood for tree
        for t in range(2,len(tree)):
            # Iterate over the tree from time t=2 to t=len(tree)
            log_likelihood += (np.log(alpha*popularity(tree,t) 
                                   + beta*root_bias(tree,t) 
                                   + tau**(novelty(tree,t)))
                                   - np.log(normalizing_factor_model_1(t,parameters)))
    print(parameters)
    print(-log_likelihood)
    return -log_likelihood

# negative log likelihood function for model_2
def negative_log_likelihood_model_2(parameters, tree_vec,parent_type_vec,type_vec):
    '''
    'parameters' are the parameters of our likelihood (derived from our attractiveness function)
        'parameters' = [alpha, beta, tau, rho]
    'data' is a list of trees in parent vector format meaning
    For a tree like
    1--2
    |  |__3
    |  |__4
    |     |__6
    |__5 

    tree_vec = [1, 2, 2, 1, 4]

    This calculates the negative log likelihood of the data given the parameters.
    The negative log likelihood is:
        l(data|parameters) = sum_{i=1}^{n} l(tree)
        Where
        l(tree) = sum_{t=2}^{|tree|} phi(tree,t) - log(Z_t)
        Where
        1) phi is the Attractiveness function
            phi(tree,t) = alpha*popularity + beta*root_bias + tau**(novelty) + rho*type_bias
        3) Z_t is the normalizing factor at time t
    '''
    
    # Initialize negative log likelihood
    log_likelihood = 0.0

    # Extract the weights from 'parameters'
    alpha = parameters[0]
    beta = parameters[1]
    tau = parameters[2]
    rho = parameters[3]

    # Loop over each tree in the dataset
    for i,tree in enumerate(tree_vec):        
        # Calculate the negative log likelihood for tree
        cur_parent_type_vec = parent_type_vec[i]
        cur_type_vec = type_vec[i]
        for t in range(2,len(tree)):
            # Iterate over the tree from time t=2 to t=len(tree)
            log_likelihood += (np.log(alpha*popularity(tree,t) 
                                   + beta*root_bias(tree,t) 
                                   + tau**(novelty(tree,t))
                                   + rho*type_bias(cur_parent_type_vec,t))
                                   - np.log(normalizing_factor_model_2(t,parameters,cur_type_vec))) #TODO: need the normalizing factor for type bias to be time-dependent
    print(parameters)
    print(-log_likelihood)
    return -log_likelihood

def callbackF(Xi):
    print('{0:4d}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f}'.format(1, Xi[0], Xi[1], Xi[2]))

# Define your optimization function
def optimize_parameters(initial_parameters, data, model_type):
    # optimization algorithm to minimize the negative log likelihood
    
    if model_type == 'model_1':
        # Using the 'L-BFGS-B' optimization algorithm
        print('{0:4s}   {1:9s}   {2:9s}   {3:9s}   {4:9s}'.format('Iter', ' alpha', ' beta', ' tau', 'NLL'))
        result = minimize(
            fun=negative_log_likelihood_model_1,
            x0=initial_parameters,
            args=(data['tree_vec'],),
            method='L-BFGS-B',
            bounds=((0,5),(0,5),(0,5)),
            # callback=callbackF
        )
    elif model_type == 'model_2':
        # Using the 'L-BFGS-B' optimization algorithm
        print('{0:4s}   {1:9s}   {2:9s}   {3:9s}   {4:9s}   {5:9s}'.format('Iter', ' alpha', ' beta', ' tau', 'rho', 'NLL'))
        result = minimize(
            fun=negative_log_likelihood_model_2,
            x0=initial_parameters,
            args=(data['tree_vec'],data['parent_type_vec'],data['type_vec']),
            method='L-BFGS-B',
            bounds=((0,5),(0,5),(0,5),(0,5)),
            # callback=callbackF
        )
    
    # Extract the optimized parameters
    optimized_parameters = result.x
    
    return optimized_parameters

def fit_model(data, model_type='model_2',num_parameters=4):
    # initial parameter values
    initial_parameters = np.random.rand(num_parameters)

    # Call the optimization function
    optimized_parameters = optimize_parameters(initial_parameters, data, model_type)

    return optimized_parameters

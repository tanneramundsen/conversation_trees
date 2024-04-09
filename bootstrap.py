import numpy as np
from sklearn.utils import resample
from model import fit_model
from parse_coarsediscourse_data import RedditComment
import pickle
from tqdm import tqdm

# Set the number of bootstrap samples and number of model parameters
num_bootstrap_samples = 30
model = 'model_1'

if model == 'model_1':
    num_parameters = 3
elif model == 'model_2':
    num_parameters = 4

# Load parent vector representation of trees
tree_vec = pickle.load(open('reddit_parent_vectors.p','rb'))
parent_type_vec = pickle.load(open('reddit_parent_type_vectors.p','rb'))
type_vec = pickle.load(open('reddit_parent_type_sum.p','rb'))
data_tree_vec = list(tree_vec.values())
data_parent_type_vec = list(parent_type_vec.values())
data_type_vec = list(type_vec.values())

# Initialize an array to store parameter estimates
bootstrap_params = np.zeros((num_bootstrap_samples, num_parameters))

# File to log parameters values each iteration


# Generate bootstrap samples and estimate parameters
with open('bootstrap_parameters_bs_10_ss_5000_model_1.txt','a',encoding='utf-8') as f:
    for i in tqdm(range(num_bootstrap_samples)):
        bootstrap_sample = resample(data_tree_vec,data_parent_type_vec,data_type_vec,n_samples=5000)
        bootstrap_data = {
            'tree_vec': bootstrap_sample[0],
            'parent_type_vec': bootstrap_sample[1],
            'type_vec': bootstrap_sample[2]
        }
        bootstrap_params[i, :] = fit_model(bootstrap_data,model,num_parameters)
        f.write(str(bootstrap_params[i, :]))
        f.write('\n')

# Analyze the distribution of parameter estimates
mean_params = np.mean(bootstrap_params, axis=0)
confidence_interval = np.percentile(bootstrap_params, [2.5, 97.5], axis=0)

# Print or use the results as needed
print("Mean Parameter Estimates:", mean_params)
print("95% Confidence Interval:", confidence_interval)

import numpy as np
from model import build_synthetic_tree
from parse_coarsediscourse_data import RedditComment
import pickle
from tqdm import tqdm

# Load parent vector representation of trees
tree_parent_vec = pickle.load(open('data/reddit_parent_vectors.p','rb'))
type_vec = pickle.load(open('data/reddit_parent_type_sum.p','rb'))

model_params = {
    'model_1': {
        'parameters': [
        0.0505, # alpha
        1.1178, # beta
        0.7009  # tau
        ],
        'type_vec': None
    },
    'model_2':  {
        'parameters': [
        0.1252, # alpha
        1.0839, # beta
        0.6955, # tau
        0.1907, #rho
        ],
        'type_vec': type_vec
    }
}

# initialize synthetic tree dict
synthetic_tree_parent_vec = {
    'model_1': {},
    'model_2': {}
}
synthetic_treelib_vec = {
    'model_1': {},
    'model_2': {}
}

for k,v in tqdm(tree_parent_vec.items()):
    for model,model_specs in model_params.items():
        if model=='model_1':
            (tree_lib_vec, parent_vec) = build_synthetic_tree(len(v),model_specs["parameters"])
        else:
            (tree_lib_vec, parent_vec) = build_synthetic_tree(len(v),model_specs["parameters"], model_specs["type_vec"][k])
        synthetic_tree_parent_vec[model][k] = parent_vec
        synthetic_treelib_vec[model][k] = tree_lib_vec

pickle.dump(synthetic_tree_parent_vec,open("coarse_discourse_synthetic_tree_parent_vec.p","wb"))
pickle.dump(synthetic_treelib_vec,open("coarse_discourse_synthetic_treelib_vec.p","wb"))
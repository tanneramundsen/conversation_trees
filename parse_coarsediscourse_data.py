from datasets import load_dataset
from treelib import Tree
import numpy as np
import pickle

class RedditComment(object):
    def __init__(self, comment_dict):
        '''
        ['title',
         'is_self_post',
         'subreddit',
         'url',
         'majority_link',
         'is_first_post',
         'majority_type',
         'id_post',
         'post_depth',
         'in_reply_to',
         'annotations',
        '''
        for k, v in comment_dict.items():
            setattr(self,k,v)


def load_data():
    ds = load_dataset("coarse_discourse")
    return ds['train']

def get_next_node_tag_counter(t,node_id):
    # Some users comment multiple times in the same Reddit Thread
    # This function iterates over the existing nodes in tree (thread)
    # and returns an integer n+1 where n is the number of times
    # 'node_id' has appeared in the tree already
    node_keys = t.nodes.keys() # list of node_ids
    n = sum(node_id in nid for nid in node_keys)
    return n+1

def get_parent_id(t,parent_id,depth):
    # We know the id of what each comment is reply to and the depth
    # of each comment. Using this info, this method returns the full
    # node id string of the correct parent node. It will be of the form
    # '<parent_id>_n' where n is equal to depth-1
    node_dict = t.nodes
    for k,v in node_dict.items():
        if parent_id in k and t.depth(v) == depth-1:
                return k
    return None


def construct_tree(ds):
    # Create dict to references to trees (treelib Tree() instances)
    # indexed by 'message_tree_id'
    all_trees = {}
    parent_vec_trees = {}
    post_type_vec = {} # dictionary of {tree_id: numpy_vec} where numpy_vec is a boolean vector
    # indicating if the parent of this node is of type {announcement, question}
    node_num_dict = {}
    node_type_dict = {}

    # ds is the data structure loaded by load_data()
    # its type is <class 'datasets.arrow_dataset.Dataset'>
    bad_egg_counter = 0
    for m in ds:
        if m["post_depth"] == -1:
            # new tree!
            tree = Tree()
            tree.create_node(m["id_post"],(m["id_post"]+"_1"),data=RedditComment(m))
            all_trees[m["title"]] = tree
            parent_vec_trees[m["title"]] = np.array([])
            post_type_vec[m["title"]] = np.array([])
            node_num_dict[m["title"]] = dict()
            node_num_dict[m["title"]][m["id_post"]+"_1"] = 1
            node_type_dict[m["title"]] = dict()
            node_type_dict[m["title"]][m["id_post"]+"_1"] = int(m["majority_type"] == 'announcement' or m["majority_type"] == 'question')
        else:
            # append to existing parent node

            # find existing tree
            tree = all_trees[m["title"]]
            node_id = m["id_post"] + "_" + str(get_next_node_tag_counter(tree,m["id_post"]))
            parent_id = get_parent_id(tree,m["in_reply_to"],m["post_depth"])
            if parent_id == None:
                bad_egg_counter = bad_egg_counter + 1
                # print(bad_egg_counter)
                # print(tree)
                # print(m)
                # print('----------------------------------------------')
                continue

            tree.create_node(m["id_post"],
                             node_id,
                             parent=parent_id,
                             data=RedditComment(m))
            
            current_tree_node_num_dict = node_num_dict[m["title"]]
            current_tree_node_type_dict = node_type_dict[m["title"]]
            if parent_id not in current_tree_node_num_dict:
                continue

            parent_num = current_tree_node_num_dict[parent_id]
            parent_type = current_tree_node_type_dict[parent_id]
            node_num_dict[m["title"]][node_id] = len(current_tree_node_num_dict)
            node_type_dict[m["title"]][node_id] = int(m["majority_type"] == 'announcement' or m["majority_type"] == 'question')

            parent_vec_trees[m["title"]] = np.append(parent_vec_trees[m["title"]],parent_num)
            post_type_vec[m["title"]] = np.append(post_type_vec[m["title"]],parent_type)
    
    # Sum up boolean vectors of node type for normalizing purposes later
    node_type_sum_dict = {k:list(v.values()) for k,v in node_type_dict.items()}

    return (all_trees, parent_vec_trees,post_type_vec,node_type_sum_dict)

def main():
    ds = load_data()
    (all_trees, parent_vec_trees,post_type_vec,node_type_sum_dict) = construct_tree(ds)
    pickle.dump(all_trees,open("coarse_discourse_trees.p","wb"))
    pickle.dump(parent_vec_trees,open("reddit_parent_vectors.p","wb"))
    pickle.dump(post_type_vec,open("reddit_parent_type_vectors.p","wb"))
    pickle.dump(node_type_sum_dict,open("reddit_parent_type_sum.p","wb"))

if __name__=="__main__":
    main()



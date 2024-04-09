import pickle
from parse_oasst_data import Message
from parse_coarsediscourse_data import RedditComment
from treelib import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

MARKERS = {'data':"o",
           'model_1':"+",
           'model_2':"v"}
COLORS = {'data':"k",
           'model_1':"b",
           'model_2':"r"}
LABELS = {'data':"data",
           'model_1':"model 1",
           'model_2':"model 2"}

def get_depth_list(all_trees):
    return [t.depth() for t in all_trees.values()]

def get_size_list(all_trees):
    return [len(t.all_nodes()) for t in all_trees.values()]

def get_degree_list(all_trees):
    return [[(1 + len(t.children(n.identifier))) for n in t.all_nodes()] for t in all_trees.values()]

def plot_series_scatterplot(series,
                            title_str,
                            x_str,
                            y_str,
                            x_log = False,
                            y_log = False,
                            ax = None,
                            marker_str = None,
                            color_str = None,
                            label_str = None):
    if ax == None:
        fig, ax = plt.subplots(figsize = (9, 6))
    if label_str == None or marker_str == None:
        ax.scatter(x=series.index.to_numpy(), y=series.to_numpy(), s=60, alpha=0.7, edgecolors="k")
    else:
        ax.scatter(x=series.index.to_numpy(), y=series.to_numpy(), s=60, alpha=0.7, edgecolors="k",color=color_str,marker=marker_str,label=label_str)
    ax.set_title(title_str)
    ax.set_xlabel(x_str)
    ax.set_ylabel(y_str)
    if x_log:
        ax.set_xscale("log")
    if y_log:
        ax.set_yscale("log")
    if ax == None:
        plt.show()
        return None

    return ax

def plot_bar(series,
        title_str,
        x_str,
        y_str,
        x_log = False,
        y_log = False):
    fig, ax = plt.subplots(figsize = (9, 6))
    ax.bar(x=series.index.to_numpy(), height=series.to_numpy().flatten())
    ax.set_title(title_str)
    ax.set_xlabel(x_str)
    ax.set_ylabel(y_str)
    ax.set_xticklabels(series.index.to_numpy(), rotation=30)
    if x_log:
        ax.set_xscale("log")
    if y_log:
        ax.set_yscale("log")
    plt.show()

def comparative_tree_stats(all_trees):
    tree_data = {}
    node_data = {}
    # Get empirical data
    empirical_data = pickle.load(open('data/coarse_discourse_trees.p','rb'))
    all_trees['data'] = empirical_data
    for model,trees in all_trees.items():
        
        depth_list = get_depth_list(trees)
        size_list = get_size_list(trees)
        # Construct pandas dataframe indexed by tree ID
        # Columns:
        #   - tree_id
        #   - depth
        #   - size
        tree_dict = {'tree_id': trees.keys(),
            'depth': depth_list,
            'size': size_list}
        tree_data[model] = pd.DataFrame(tree_dict)

        # Construct pandas dataframe indexed by message_id (nodes of tree)
        node_list = []
        depth_list = []
        degree_list = []
        for t in trees.values():
            for n in t.all_nodes():
                depth_list.append(t.depth(n.identifier))
                degree_list.append(len(t.children(n.identifier)))
                node_list.append(n)

        message_ids_list = [n.identifier for n in node_list]
        node_dict = {'message_id': message_ids_list,
                    'depth': depth_list,
                    'degree': degree_list,
        }
        node_data[model] = pd.DataFrame(node_dict) 

    # Plot metrics
    fig_1 = plt.figure()
    ax1 = fig_1.add_subplot(111)
    for model,tree_df in tree_data.items():
        # Average Depth by Size
        size_group = tree_df.groupby('size')
        depth_by_size = size_group[["depth"]]
        avg_depth_by_size = depth_by_size.mean()
        ax1 = plot_series_scatterplot(avg_depth_by_size,
                                'Size vs. Average Depth',
                                'Size',
                                'Average Depth',
                                x_log = True,
                                y_log = True,
                                ax = ax1,
                                marker_str = MARKERS[model],
                                color_str = COLORS[model],
                                label_str = LABELS[model])
    ax1.legend(loc='upper left')
    plt.show()
    # Proportion of node degree
    fig_2 = plt.figure()
    ax2 = fig_2.add_subplot(111)
    for model,node_df in node_data.items():
        degree_group = node_df.groupby('degree')
        count_degree_group = degree_group.size()
        degree_proportion = count_degree_group.apply(lambda x: x / len(node_df))
        ax2 = plot_series_scatterplot(degree_proportion,
                                'Distribution of Degree',
                                'Degree',
                                'Fraction',
                                x_log = True,
                                y_log = True,
                                ax = ax2,
                                marker_str = MARKERS[model],
                                color_str = COLORS[model],
                                label_str = LABELS[model])
        # print('Mean degree:')
        # print(node_df[["degree"]].mean())
        # print('Std Dev degree:')
        # print(node_df[["degree"]].std())
    ax2.legend(loc='upper right')
    plt.show()

def main():
    args = sys.argv[1:]
    print(args[0])
    f = open(args[0],'rb')
    f.seek(0)
    all_trees = pickle.load(f)
    if isinstance(list(all_trees.values())[0], dict):
        comparative_tree_stats(all_trees)
        return
    
    depth_list = get_depth_list(all_trees)
    size_list = get_size_list(all_trees)

    # Construct pandas dataframe indexed by tree ID
    # Columns:
    #   - tree_id
    #   - depth
    #   - size
    tree_dict = {'tree_id': all_trees.keys(),
         'depth': depth_list,
         'size': size_list}
    tree_df = pd.DataFrame(tree_dict)

    # Construct pandas dataframe indexed by message_id (nodes of tree)
    node_list = []
    depth_list = []
    degree_list = []
    for t in all_trees.values():
        for n in t.all_nodes():
            depth_list.append(t.depth(n.identifier))
            degree_list.append(len(t.children(n.identifier)))
            node_list.append(n)

    message_ids_list = [n.identifier for n in node_list]
    if args[0] == 'data/coarse_discourse_trees.p':
        message_tree_ids_list = [n.data.title for n in node_list]
        majority_type_list = [n.data.majority_type for n in node_list]
        node_dict = {'message_id': message_ids_list,
                    'depth': depth_list,
                    'degree': degree_list,
                    'majority_type': majority_type_list,
                    'message_tree_id': message_tree_ids_list
        }
    else:
        message_tree_ids_list = [n.data.message_tree_id for n in node_list]
        created_date_list = [n.data.created_date for n in node_list]

        node_dict = {'message_id': message_ids_list,
                    'depth': depth_list,
                    'degree': degree_list,
                    'created_date': created_date_list,
                    'message_tree_id': message_tree_ids_list
        }

    node_df = pd.DataFrame(node_dict)

    # Plot metrics

    # Proportion of trees by size
    size_group = tree_df.groupby('size')
    count_size_group = size_group.size()
    size_proportion = count_size_group.apply(lambda x: x / len(tree_df))
    plot_series_scatterplot(size_proportion,
                            'Distribution of Size',
                            'Size',
                            'Fraction',
                            x_log = True,
                            y_log = True)
    print('Mean size:')
    print(tree_df[["size"]].mean())
    print('Std Dev size:')
    print(tree_df[["size"]].std())

    # Proprortion of trees by depth
    depth_group = tree_df.groupby('depth')
    count_depth_group = depth_group.size()
    depth_proportion = count_depth_group.apply(lambda x: x / len(tree_df))
    plot_series_scatterplot(depth_proportion,
                            'Distribution of Depth',
                            'Depth',
                            'Fraction',
                            x_log = True,
                            y_log = True)
    print('Mean depth:')
    print(tree_df[["depth"]].mean())
    print('Std Dev depth:')
    print(tree_df[["depth"]].std())

    # Average Depth by Size
    depth_by_size = size_group[["depth"]]
    avg_depth_by_size = depth_by_size.mean()
    plot_series_scatterplot(avg_depth_by_size,
                            'Size vs. Average Depth',
                            'Size',
                            'Average Depth',
                            x_log = True,
                            y_log = True)

    # Proportion of node degree
    degree_group = node_df.groupby('degree')
    count_degree_group = degree_group.size()
    degree_proportion = count_degree_group.apply(lambda x: x / len(node_df))
    plot_series_scatterplot(degree_proportion,
                            'Distribution of Degree',
                            'Degree',
                            'Fraction',
                            x_log = True,
                            y_log = True)
    print('Mean degree:')
    print(node_df[["degree"]].mean())
    print('Std Dev degree:')
    print(node_df[["degree"]].std())

    if args[0] == 'data/coarse_discourse_trees.p':
        majority_type_group = node_df.groupby('majority_type')
        # proportion of majority type
        count_majority_type_group = majority_type_group.size()
        plot_bar(count_majority_type_group[1:],
             'Message Type Count',
             'Message Type',
             'Count')
        
        # 'degree' by 'majority_type'
        degree_by_majority_type = majority_type_group[["degree"]]
        avg_degree_by_majority_type = degree_by_majority_type.mean()
        plot_bar(avg_degree_by_majority_type[1:],
                 'Average Degree by Message Type',
                 'Message Type',
                 'Average Degree')


    # Degree by node level (depth)
    # degree_by_depth = degree_group[["depth"]]

if __name__=="__main__":
    main()

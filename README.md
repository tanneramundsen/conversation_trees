# conversation_tree_branching
Final project for EN.625.721 on the theory and applications of branching processes and time series analysis. 

## Requirements
Run

`pip3 install -r requirements.txt`

## Dataset
The first dataset for this project comes from [Hugging Face Coarse Discourse](https://huggingface.co/datasets/coarse_discourse). This dataset contains comment threads from Reddit. The comment were annotated by paid crowdsources. The threads  were sampled randomly from various subreddits and the rows appear in time order.

The second dataset for this project comes from [Open Assistant Conversation Dataset (OASST1)](https://huggingface.co/datasets/OpenAssistant/oasst1). Open Assistant summarizes its dataset as "a human-generated, human-annotated assistant-style conversation corpus consisting of 161,443 messages in 35 different languages, annotated with 461,292 quality ratings, resulting in over 10,000 fully annotated conversation trees. The corpus is a product of a worldwide crowd-sourcing effort involving over 13,500 volunteers." 

The data is organized into message trees. Message trees are data structures where each node represents a piece of dialogue. In the case of Open Assistant, these conversations are between users and chatbot assistants. The root of each message tree is an initial prompt. Deeper layers consist of replies to this intial prompt - alternating between "prompter" and "assistant". 

## Data loading
You can load the oasst1 dataset from HuggingFace `datasets` by running

`python3 parse_oasst_data.py`

`python3 parse_coarsediscourse_data.py`

Which will create a `pickle` file named `open_assistant_tree.p` or `coarse_discourse_trees.p` containing a single Python dictionary named `all_trees`. `all_trees` is indexed by thread id.

## Analysis of Data
Run 

`python3 get_tree_stats.py data/<dataset_pickle_name.p>`

which prints summary statistics of the threads and makes plots. If you are interested in plotting the descriptive statistics of the synthetic trees, use 'data/coarse_discourse_synthetic_treelib_vec.p'

## Fitting to Model
Run

`python3 bootstrap.py`

which at time of writing is set to fit [Model (1)](https://arxiv.org/abs/1203.0652) to the Reddit data (Coarse Discourse).

## Synthetic Tree Generation
Run

`python3 generate_trees.py`

which at time of writing is set to create a file called 'data/coarse_discourse_synthetic_treelib_vec.p' of data created from Model 1 and Mode 2. It is set to read from 'data/reddit_parent_vectors.p' and 'data/reddit_parent_type_sum.p'
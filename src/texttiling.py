
# importing necessary libraries and modules
import math
import spacy
import networkx as nx
import community as community_louvain
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel
import torch
from scipy.spatial.distance import cosine
from bert_score import score

def find_overlap(vector1, vector2):
    min_v1, max_v1 = min(vector1), max(vector1)
    min_v2, max_v2 = min(vector2), max(vector2)

    one_in_two = [num for num in vector1 if min_v2 <= num <= max_v2]
    two_in_one = [num for num in vector2 if min_v1 <= num <= max_v1]
    overlap = one_in_two+two_in_one

    return overlap, len(one_in_two), len(two_in_one)

def get_similarity_scores(vec_a, vec_b, model):
    """
    This function calculates the similarity scores between pairs of text chunks using the specified model.
    It supports several types of models, including BERT, sequence matcher, Jaccard index, and transformers from HuggingFace.

    Parameters:
    - vec_a, vec_b: Lists of text chunks to be compared. They must have the same length, and the comparison is made
      between corresponding pairs (i.e., vec_a[i] is compared with vec_b[i]).
    - model: A string indicating the type of model to be used for the comparison. Current valid values are "bert", "seqmatch", "jaccard", 
      and the name of any transformer available from HuggingFace.

    Returns:
    - similarities: A list of similarity scores for each pair of text chunks. The score ranges from 0 (no similarity) to 1 (identical).
    """
    vec_a = [x.lower() for x in vec_a]  # convert vector a to lowercase
    vec_b = [x.lower() for x in vec_b]  # convert vector b to lowercase

    if model == "bertscore":
        # BERTScore returns three values: Precision, Recall, and F1 Score
        # Here we use F1 Score as the similarity measure
        _, _, f1_score = score(vec_a, vec_b, lang='pt', model_type='xlm-roberta-base')

        # f1_score is a tensor with the F1 score for each pair of sentences.
        # Since we only have one pair, we take the first (and only) element.
        similarities = [f1_score[i].item() for i in range(len(vec_a))]
    else: 
        # any transformer chosen from HuggingFace can be used here
        similarities = []       
        tokenizer = AutoTokenizer.from_pretrained(model)
        model = AutoModel.from_pretrained(model)
        embeddings_a = []
        embeddings_b = []

        # Compute embeddings for each string in the lists
        for i in range(len(vec_a)):
            a_tokens = tokenizer(vec_a[i], padding=True, truncation=True, max_length=256, return_tensors='pt')
            b_tokens = tokenizer(vec_b[i], padding=True, truncation=True, max_length=256, return_tensors='pt')

            with torch.no_grad():
                embeddings_a.append(model(**a_tokens).last_hidden_state.mean(dim=1))
                embeddings_b.append(model(**b_tokens).last_hidden_state.mean(dim=1))

        # Compute cosine similarity for each pair of strings
        similarities = []
        for i in range(len(embeddings_a)):
            similarity = 1 - cosine(embeddings_a[i][0], embeddings_b[i][0])
            similarity = (similarity + 1) / 2
            similarity = max(0.0, min(1.0, similarity))
            similarities.append(similarity)
    return similarities

def compact_clusters(clusters):
    """
    This function takes a list of clusters and compacts them, eliminating range overlaps.
    It does this by iteratively merging overlapping clusters together until there are no more overlaps.
    The approach taken is to minimize the number of elements that need to be moved during the merge.

    Parameters:
    - clusters: a list of lists, where each sublist represents a cluster of elements.

    Returns:
    - compact_clusters: a list of compacted clusters. The clusters are sorted, and the elements within each cluster are also sorted.
    """
    compact_clusters = []
    while len(clusters):
        curr_cl = clusters.pop(0)
        if not curr_cl:
            pass
        for i in range(len(clusters)):
            target_cl = clusters[i]
            # find_overlap() returns the range overlaps and number of overlaps between two clusters
            overlap, n_1_in_2, n_2_in_1 = find_overlap(target_cl, curr_cl)
            if overlap:
                # The code block here decides which cluster to merge the overlapping elements into.
                # It aims to minimize the amount of element transfer. If it's equally easy to merge into both,
                # it merges into the current cluster.
                if n_1_in_2 < n_2_in_1 or n_2_in_1 == 0:
                    curr_cl.extend(overlap)
                    curr_cl = list(set(curr_cl))
                    clusters[i] = list(set(target_cl)-set(overlap))
                else:
                    target_cl.extend(overlap)
                    target_cl = list(set(target_cl))
                    curr_cl = list(set(curr_cl)-set(overlap))
                if not curr_cl:
                    break
        # After examining all clusters, if the current cluster still has elements, it's added to the compacted list.
        if len(curr_cl):
            # remove any possible duplicates
            compact_clusters.append(list(set(curr_cl)))

    compact_clusters

    for cl in compact_clusters:
        cl.sort()
    compact_clusters.sort()
    return compact_clusters

def create_similarity_graph(sentences, K, model):
    """
    This function creates a graph of sentence similarities given a list of input sentences.
    Each sentence is connected with the following K sentences in the list. The similarity between each pair of sentences
    is calculated using the provided model, and this similarity is then used to weight the edge between the sentences in the graph.
    
    Parameters:
    - sentences: a list of sentences (strings) for which the similarity graph is to be created.
    - K: an integer indicating the number of following sentences to be considered for each sentence.
    - model: a string indicating the model to be used for calculating sentence similarity.
    
    Returns:
    - result: a list of weighted edges in the graph. Each edge is represented by a list of three elements: 
      the index of the first sentence, the index of the second sentence, and the weight of the edge.
    """
    result = []
    couples = []

    # The outer loop iterates over all sentences.
    for i in range(len(sentences)):
        l = 0
        # The inner loop iterates over K sentences following the sentence i.
        for j in range(i+1, min(i+1+K, len(sentences))):
            # Collecting pairs of sentences and assign a decreasing weight to the edge connecting them.
            # This reflects the intuition that closer sentences are more likely to be similar.
            couples.append((sentences[i], sentences[j]))
            result.append([i, j, math.exp(-l/2)]) # weight decreases as we move further away
            l += 1

    # Couples are split into two separate lists which are fed to the similarity function
    a, b = zip(*couples)
    similarities = get_similarity_scores(a, b, model)
    # The similarity score for each pair of sentences is incorporated into the edge weight.
    for i, s in enumerate(similarities):
        result[i][2] *= s

    # The final result is a list of weighted edges in the similarity graph.
    return result

def create_tiles(text, model='jvanhoof/all-MiniLM-L6-multilingual-v2-en-es-pt-pt-br-v2'):
    """
    This function segments a text into sentences, and groups similar sentences into 'tiles' or clusters. 
    Each tile represents a section of the text where the sentences are considered similar based on the similarity model used. 
    The function then removes overlaps from these tiles and returns the text tiling in the form [0, th1, th2.... thN, len(sentences)-1] 
    
    Parameters:
    - doc_path: a string that is the path to the text document to be read.
    - model: a string indicating the similarity model to use. Default is 'bert'.
    
    Returns:
    - a vector containing the list of thresholds representing the tiling for input document.
    """
    text = text.replace('\n', ' ')

    # Process the text to get a Doc object
    nlp = spacy.load("pt_core_news_sm")
    doc = nlp(text)
    # Extract sentences from the Doc object
    sentences = [sent.text for sent in doc.sents]
    
    # Create a similarity graph based on the extracted sentences
    graph = create_similarity_graph(sentences, 5, model=model)

    # Initialize a Graph object from Networkx
    G = nx.Graph()
    for node in graph:
        G.add_edge(node[0], node[1], weight=node[2])

    # Partition the graph into communities using the Louvain method
    partition = community_louvain.best_partition(G, resolution=1., weight='weight', randomize=False)

    # Organize the sentences into their respective communities (tiles)
    tiles = defaultdict(list)
    for k, v in partition.items():
        tiles[v].append(k)

    # Convert the defaultdict to a list and sort it
    tiles = list(tiles.values())
    tiles.sort()

    # Compact the clusters to remove range overlaps
    tiles = compact_clusters(tiles)

    # Generate the text tiling in the form [0, th1, th2.... thN, len(sentences)-1] 
    tiles_idxs = [c[0] for c in tiles]+[tiles[-1][-1]]
    tiled_text = ['\n'.join(sentences[tiles_idxs[idx] : tiles_idxs[idx+1]]) for idx in range(len(tiles_idxs) - 1)]
    return tiles_idxs, tiled_text



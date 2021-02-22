
from deg_project.general import general_utilies
import networkx as nx
import csv
def disjoint_split(): 
    seq_list = general_utilies.pd.read_csv(general_utilies.seq_PATH)["seq"].values.tolist()
    hash_map = {}
    k=20 #k-mer length 
    l=110 #seqence fixed length
    G = nx.Graph()
    G.add_nodes_from(range(len(seq_list)))

    #passing all k-mer in the seqences and create the edges accordingly 
    for i,seq in enumerate(seq_list):
        for j in range(l-k+1):
            kmer=seq[j:j+k]
            if kmer in hash_map:
                hash_map[kmer].add(i)
                edges_list = [(i,m) for m in hash_map[kmer]]
                G.add_edges_from(edges_list)
            else:
                hash_map[kmer] = {i}

    #create the connected components
    connected_components_list = list(nx.connected_components(G))

    #create the disjoint  train, validation, and test sets
    # This may not deliver the same ids as we provided, since we could not reproduce the random pick 
    train = list(connected_components_list[0]) #the first connected component contains 67280 sequences
    test = []
    i = 1
    while ((len(test)+len(connected_components_list[i]))<=10000):
        test = test + list(connected_components_list[i])
        i+=1
    validation = []
    for connected_component in connected_components_list[i:]:
        validation = validation + list(connected_component)

    #save the ids in a csv file
    output_csv = [[id] for id in train]
    for i in range(len(validation)):
        output_csv[i].append(validation[i])
    for i in range(len(test)):
        output_csv[i].append(test[i])
    output_csv = [["train_ids", "validation_ids", "test_ids"]]+ output_csv
    with open(general_utilies.files_dir+'split_to_train_validation_test_disjoint_sets_ids.csv', 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerows(output_csv)
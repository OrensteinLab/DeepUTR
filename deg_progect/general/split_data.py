import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, "/home/u30614/deg_project/")
from deg_project.general import utilies

seq_list = utilies.pd.read_csv(utilies.seq_PATH)["seq"].values.tolist()
hash_map = {}
k=15
l=110
for i,seq in enumerate(seq_list):
    for j in range(l-k+1):
        kmer=seq[j:j+k]
        if kmer in hash_map:
            hash_map[kmer].add(i)
        else:
            hash_map[kmer] = {i}

hash_map_counts = {k: len(v) for k, v in hash_map.items()}
hash_map_counts = list(hash_map_counts.values())
hash_map_counts.sort()

hash_map_counts_above_5 = [hash_map_count for hash_map_count in hash_map_counts if hash_map_count>5]
import matplotlib.pyplot as plt
plt.plot(range(len(hash_map_counts_above_5)), hash_map_counts_above_5)
plt.show()
# %%
plt.hist([hash_map_count for hash_map_count in hash_map_counts if hash_map_count>25], bins=350)
plt.show()

# %%

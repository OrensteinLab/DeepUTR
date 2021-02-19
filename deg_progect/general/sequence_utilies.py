import numpy as np
from itertools import product
import logomaker

###############one hot######################
# one hot encoding function
def one_hot_encoding(seq):
    mapping = dict(zip("ACGT", range(4)))    
    seq2 = [mapping[i] for i in seq]
    #return seq2  # -use if using embbiding TODO support this from run if need 
    return np.eye(4)[seq2].astype('uint8')

###############kmer count######################
#kmer utilities functions 
def CountKmer (seq, k):
    kFreq = {}
    for i in range (0, len(seq)-k+1):
        kmer = seq [i:i+k]
        if kmer in kFreq:
            kFreq [kmer] += 1
        else:
            kFreq [kmer] = 1
    return kFreq

def retutnAllKmers (min_kmer_length, max_kmer_length):
    kmer_list = []
    for i in range (min_kmer_length, max_kmer_length+1):
        kmer_list = kmer_list + [''.join(c) for c in product('ACGT', repeat=i)]
    return kmer_list

def createFeturesVector (allKmers, seqkMerCounter):
    AllKmersSize = len(allKmers)
    KmerCounterArray = np.zeros((AllKmersSize,1))
    for i in range (0, AllKmersSize):
        if allKmers[i] in seqkMerCounter:
              KmerCounterArray [i] = seqkMerCounter[allKmers[i]]
    return KmerCounterArray

def createFeturesVectorsForAllSeq (allKmers, min_kmer_length, max_kmer_length, sequences):
    num_of_kmers = len(allKmers)
    num_of_sequences = len(sequences)
    FeturesVectorsOfAllSeq = np.zeros((num_of_sequences, num_of_kmers, 1), dtype='int8')
    for i in range(num_of_sequences): 
        seq = sequences [i]
        seqkMerCounter = {}
        for j in range (min_kmer_length, max_kmer_length+1):
            seqkMerCounter = {**seqkMerCounter, **CountKmer(seq, j)}
        FeturesVectorsOfAllSeq [i] = createFeturesVector (allKmers, seqkMerCounter)
    return FeturesVectorsOfAllSeq


###############################create Logo object#######################################
# create Logo object
def create_DNA_logo (PWM_df, secondary_color=False, figsize=(10, 2.5), labelpad=-1, ax=None):
    if(secondary_color):
        color_scheme='NajafabadiEtAl2017'
    else:
        color_scheme='classic'

    IG_logo = logomaker.Logo(PWM_df,
                            shade_below=.5,
                            fade_below=.5,
                            color_scheme=color_scheme,
                            font_name='Arial Rounded MT Bold',
                            ax=ax,
                            figsize=figsize)

    IG_logo.style_spines(visible=False)
    IG_logo.style_spines(spines=['left', 'bottom'], visible=True)
    IG_logo.style_xticks(rotation=90, fmt='%d', anchor=0)

    # style using Axes methods
    IG_logo.ax.set_ylabel("IG", labelpad=labelpad)
    #IG_logo.ax.set_xlabel(string)
    IG_logo.ax.xaxis.set_ticks_position('none')
    IG_logo.ax.xaxis.set_tick_params('both')
    IG_logo.ax.set_xticklabels([])

    return IG_logo

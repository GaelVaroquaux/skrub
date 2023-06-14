"""
A fast (but not very good) nearest neighbor on strings.

It uses a really poor min-hash implementation to linear-time approximate
nearest neighbor matching on strings.
"""
import pandas as pd


import numpy as np

# Numbers chosen to give good entropy, modulo 256 (from -128 to 128)
MODULOS = [13, 97, -37, 67, -43, 83, -79, 107, -113, 41, -53, 7, 71, -83,
           151, -73, 89, -47, 107, -109]

# TODO: for really big array, we should consider making this by
# chunks of rows, optionally parallel
def cheap_ngram_minhash(strings, n_hashes=8):
    """ A very fast but quite poor 3-gram minhash index, for string similarity.

    Parameters
    ==========
    strings: {pd.Series, ndarray, or list of strings}
    """
    if isinstance(strings, pd.Series):
        # Cast everything to non unicode string (creates collisions, hopefully
        # desirable)
        strings = strings.str.encode('latin1', errors='replace')
        # Convert to a numpy array of bytes (zero padded)
        strings = strings.array.astype(bytes)
    elif isinstance(strings, np.ndarray):
        strings = np.astype(strings, bytes)
    else:
        strings = np.array(strings, dtype=bytes)
    n_samples = strings.shape[0]

    array = np.frombuffer(strings, dtype="int8")
    # Reshape to have one string per line
    array = array.reshape((n_samples, -1))

    # Super cheap ngram hash (use a modulo to vary the hashes) and minhash
    # (technicall a min-max hash).

    ngrams = np.lib.stride_tricks.sliding_window_view(array, window_shape=3, axis=1)
    # using as a dtype int8 creates a modulo and thus hash collisions
    # In practice, below, we implement a tiny linear congruential
    # generator.
    # TODO: explore using product and not only addition
    hashes = ngrams.sum(axis=-1, dtype='int8')
    # Save memory
    del ngrams, array, strings

    modulos = MODULOS[:n_hashes]
    minmaxhash = np.empty(shape=(n_samples, 2 * len(modulos)), dtype='int8')
    for i, modulo in enumerate(modulos):
        # TODO: explore using product and not only addition
        this_hashes = np.add(hashes, modulo, dtype='int8')
        minmaxhash[:, 2 * i] = this_hashes.min(axis=-1)
        minmaxhash[:, 2 * i + 1] = this_hashes.max(axis=-1)
    return minmaxhash

## TODO:
# - Unit tests:
#   - a test case that checks that exact matchings always give
#     100% hash collisions
# - Do intensive benchmarking to:
#   - Tune the parameters of the Linear Congruential Generator https://en.wikipedia.org/wiki/Linear_congruential_generator#c_%E2%89%A0_0
#     Maybe figure of merit here should be how well the Jaccard is
#     approximated
#   - Choose the default size of ngrams, maybe as a function of the dtype
#     of the string array (which tells us the length of the longuest
#     string, in which case we need to return the chosen value, to be
#     able to have identical train and test
#   - Do performance optimizations, look also at memory consumption
# - Consider API (objects, functions?) for best match (or good-enough
#   match), multiple queries, screening
# - Merge with skrub/_minhash_encoder.py (requires consider maybe a
#   common signature)

#################################################
# Examples

# ----------------------------------------------
# Simple test case that works well
from skrub import datasets
data = datasets.fetch_employee_salaries()
df = data.X
# We use real data, but only with unique values
strings = pd.Series(df['employee_position_title'].unique())

# Query goes as such:
minmaxhash = cheap_ngram_minhash(strings)
query_string = 'Land Survey Superviser'
print(f'Query string: {query_string}')
query = cheap_ngram_minhash([query_string, ])

match_idx = np.argmax((minmaxhash == query).sum(axis=-1))
print(f'Found {strings[match_idx]}')

# Idea: Use a fast_dict to lookup really fast. But will look up only
# exact matches

# ----------------------------------------------
# Harder real-life case
data = datasets.get_ken_embeddings()
strings = data['Entity'].str[1:-1]

minmaxhash = cheap_ngram_minhash(strings)
# Consider 4grams?
query = cheap_ngram_minhash(['Horatio, Mississippi', ])

nb_hashes_diff = (minmaxhash != query).sum(axis=-1)
n_matches_to_keep = 10
selection = np.argpartition(nb_hashes_diff,
                            n_matches_to_keep)[:n_matches_to_keep]
print(strings.iloc[selection].tolist())

# ----------------------------------------------
# ' Mississippi' is hard to match to '_Mississippi'.
# This is probably due to the number of repeated n_gram in "Mississippi"
print(np.mean(cheap_ngram_minhash(['_Mississippi', ], n_hashes=20)
      == cheap_ngram_minhash([' Mississippi', ], n_hashes=20)))
# The above gives only 0.45, while we expect much more due to a high
# Jaccard

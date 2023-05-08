import math
from collections import Counter
from fractions import Fraction

def tokenize(text):
    """
    Tokenize a given text by splitting on whitespace.
    
    Args:
    text (str): The input text.

    Returns:
    list of str: The list of tokens.
    """
    return text.split()

def ngram_counts(tokens, n):
    """
    Count the n-grams in a list of tokens.

    Args:
    tokens (list of str): The list of tokens.
    n (int): The length of n-grams to consider.

    Returns:
    Counter: A Counter object containing n-gram counts.
    """
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))

def clipped_ngram_count(candidate_ngrams, reference_ngrams):
    """
    Calculate the clipped n-gram count for the candidate and reference n-grams.

    Args:
    candidate_ngrams (Counter): The candidate n-gram counts.
    reference_ngrams (Counter): The reference n-gram counts.

    Returns:
    dict: A dictionary containing clipped n-gram counts.
    """
    clipped_counts = {
        ngram: min(candidate_ngrams[ngram], reference_ngrams[ngram])
        for ngram in candidate_ngrams
    }
    return clipped_counts

def modified_precision(candidate, reference, n):
    """
    Calculate the modified precision for the given candidate and reference texts.

    Args:
    candidate (list of str): The list of candidate tokens.
    reference (list of str): The list of reference tokens.
    n (int): The length of n-grams to consider.

    Returns:
    float: The modified precision.
    """
    candidate_ngrams = ngram_counts(candidate, n)
    reference_ngrams = ngram_counts(reference, n)
    clipped_counts = clipped_ngram_count(candidate_ngrams, reference_ngrams)
    clipped_count_sum = sum(clipped_counts.values())
    total_count_sum = sum(candidate_ngrams.values())

    if clipped_count_sum == 0:
        return 0
    return Fraction(clipped_count_sum, total_count_sum)

def closest_length_ratio(candidate_length, reference_lengths):
    """
    Calculate the closest length ratio between the candidate and reference texts.

    Args:
    candidate_length (int): The length of the candidate text.
    reference_lengths (list of int): A list of lengths for each reference text.

    Returns:
    float: The closest length ratio.
    """
    ref_diffs = [abs(candidate_length - ref_length) for ref_length in reference_lengths]
    closest_length = reference_lengths[ref_diffs.index(min(ref_diffs))]
    return candidate_length / closest_length

def brevity_penalty(candidate_length, reference_lengths):
    """
    Calculate the brevity penalty for the given candidate and reference texts.

    Args:
    candidate_length (int): The length of the candidate text.
    reference_lengths (list of int): A list of lengths for each reference text.

    Returns:
    float: The brevity penalty.
    """
    c = candidate_length
    r = closest_length_ratio(candidate_length, reference_lengths)
    
    if c > r:
        return 1
    else:
        return math.exp(1 - r)

def bleu_score(candidate, references, max_n=4, weights=None):
    """
    Calculate BLEU score given a candidate translation and a list of reference translations.

    Args:
    candidate (str): The candidate translation string.
    references (list of str): A list of reference translation strings.
    max_n (int, optional): The maximum length of n-grams to consider(up to 4). Defaults to 4.
    weights (list of float, optional): The weights for the modified precision scores of each n-gram length. Defaults to uniform weights.

    Returns:
    float: The BLEU score (0-1) for the candidate translation.
    """
    if weights is None:
        weights = [1 / max_n] * max_n

    candidate_tokens = tokenize(candidate)
    reference_tokens_list = [tokenize(ref) for ref in references]

    candidate_length = len(candidate_tokens)
    reference_lengths = [len(ref_tokens) for ref_tokens in reference_tokens_list]

    bp = brevity_penalty(candidate_length, reference_lengths)

    precisions = [
        modified_precision(candidate_tokens, ref_tokens, n)
        for n in range(1, max_n + 1)
        for ref_tokens in reference_tokens_list
    ]

    weighted_precisions = [w * math.log(p) for w, p in zip(weights, precisions)]
    bleu = bp * math.exp(sum(weighted_precisions))

    return bleu

def rouge_score(candidate, references, n=2):
    """
    Calculate ROUGE-N score given a candidate summary and a list of reference summaries.
    
    Args:
    candidate (str): The candidate summary string.
    references (list of str): A list of reference summary strings.
    n (int, optional): The length of n-grams to consider. Defaults to 2 (bigrams).

    Returns:
    float: The ROUGE-N score (0-1) for the candidate summary.
    """
    candidate_tokens = tokenize(candidate)
    reference_tokens_list = [tokenize(ref) for ref in references]

    candidate_ngrams = ngram_counts(candidate_tokens, n)
    reference_ngrams_list = [ngram_counts(ref_tokens, n) for ref_tokens in reference_tokens_list]

    max_clipped_counts = Counter()
    for ref_ngrams in reference_ngrams_list:
        clipped_counts = clipped_ngram_count(candidate_ngrams, ref_ngrams)
        max_clipped_counts = max(max_clipped_counts, clipped_counts, key=lambda c: sum(c.values()))

    rouge_numerator = sum(max_clipped_counts.values())
    rouge_denominator = sum(candidate_ngrams.values())

    return rouge_numerator / rouge_denominator


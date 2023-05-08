import math
from collections import Counter
from fractions import Fraction

# Tokenization
def tokenize(text):
    return text.split()

# N-gram counting
def ngram_counts(tokens, n):
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))

# BLEU
def clipped_ngram_count(candidate_ngrams, reference_ngrams):
    clipped_counts = {
        ngram: min(candidate_ngrams[ngram], reference_ngrams[ngram])
        for ngram in candidate_ngrams
    }
    return clipped_counts

def modified_precision(candidate, reference, n):
    candidate_ngrams = ngram_counts(candidate, n)
    reference_ngrams = ngram_counts(reference, n)
    clipped_counts = clipped_ngram_count(candidate_ngrams, reference_ngrams)
    clipped_count_sum = sum(clipped_counts.values())
    total_count_sum = sum(candidate_ngrams.values())

    if clipped_count_sum == 0:
        return 0
    return Fraction(clipped_count_sum, total_count_sum)

def closest_length_ratio(candidate_length, reference_lengths):
    ref_diffs = [abs(candidate_length - ref_length) for ref_length in reference_lengths]
    closest_length = reference_lengths[ref_diffs.index(min(ref_diffs))]
    return candidate_length / closest_length

def brevity_penalty(candidate_length, reference_lengths):
    c = candidate_length
    r = closest_length_ratio(candidate_length, reference_lengths)
    
    if c > r:
        return 1
    else:
        return math.exp(1 - r)

def bleu_score(candidate, references, max_n=4, weights=None):
    if weights is None:
        weights = [1 / max_n] * max_n

    candidate_tokens = tokenize(candidate)
    reference_tokens = [tokenize(ref) for ref in references]

    candidate_length = len(candidate_tokens)
    reference_lengths = [len(ref_tokens) for ref_tokens in reference_tokens]

    precisions = [
        modified_precision(candidate_tokens, ref_tokens, n)
        for n in range(1, max_n + 1)
        for ref_tokens in reference_tokens
    ]

    weighted_log_precisions = [
        weight * math.log(precision) if precision > 0 else 0
        for weight, precision in zip(weights, precisions)
    ]

    score = brevity_penalty(candidate_length, reference_lengths) * math.exp(sum(weighted_log_precisions))
    return score

# ROUGE
def rouge_n(candidate, reference, n):
    candidate_tokens = tokenize(candidate)
    reference_tokens = tokenize(reference)

    candidate_ngrams = ngram_counts(candidate_tokens, n)
    reference_ngrams = ngram_counts(reference_tokens, n)

    overlapping_ngrams = sum(min(candidate_ngrams[ngram], reference_ngrams[ngram]) for ngram in candidate_ngrams)
    total_reference_ngrams = sum(reference_ngrams.values())

    if total_reference_ngrams == 0:
        return 0

    return overlapping_ngrams / total_reference_ngrams

def rouge_score(candidate, references, n=2):
    scores = [rouge_n(candidate, reference, n) for reference in references]
    return max(scores)

from collections import Counter
import numpy as np
from nltk import ngrams
import torch

def simple_count(tokens, n):
    return Counter(ngrams(tokens, n))

def count_clip(candidate, reference_list, n):
    cnt_ca = simple_count(candidate, n)
    temp = dict()

    for ref in reference_list:
        cnt_ref = simple_count(ref, n)
        for n_gram in cnt_ref:
            if n_gram in temp:
                temp[n_gram] = max(cnt_ref[n_gram], temp[n_gram])
            else:
                temp[n_gram] = cnt_ref[n_gram]

    return {
        n_gram: min(cnt_ca.get(n_gram, 0), temp.get(n_gram, 0)) for n_gram in cnt_ca
    }

def precision(candidate, reference_list, n):
    clip = count_clip(candidate, reference_list, n)
    total_clip = sum(clip.values())

    ct = simple_count(candidate, n)
    total_ct = sum(ct.values())

    if total_ct == 0:
        total_ct = 1

    return (total_clip, total_ct)

def closest_ref_length(candidate, reference_list):
    ca_len = len(candidate)
    ref_lens = (len(ref) for ref in reference_list)
    closest_ref_len = min(ref_lens, key=lambda ref_len: (abs(ref_len - ca_len), ref_len))
    return closest_ref_len

def brevity_penalty(candidate, reference_list):
    ca_len = len(candidate)
    ref_len = closest_ref_length(candidate, reference_list)

    if ca_len > ref_len:
        return 1
    elif ca_len == 0:
        return 0
    else:
        return np.exp(1 - ref_len/ca_len)

def bleu_score(candidate, reference_list, weights=[0.25, 0.25, 0.25, 0.25]):
    """
    ret: [bleu1, bleu2, bleu3, bleu4, bleu]
    """
    score_list = [0.0, 0.0, 0.0, 0.0]
    p_numerators = [0.0, 0.0, 0.0, 0.0]
    p_denominators = [0.0, 0.0, 0.0, 0.0]
    for cand, ref in zip(candidate, reference_list):
        bp = brevity_penalty(cand, ref)
        for i, _ in enumerate(weights, start=1):
            p_i = precision(cand, ref, n=i)
            p_numerators[i-1] += p_i[0]
            p_denominators[i-1] += p_i[1]

    for i, _ in enumerate(weights):
        score_list[i] = p_numerators[i] / p_denominators[i]

    score = np.sum([w_i * np.log(p_i) if p_i != 0 else 0 for w_i, p_i in zip(weights, score_list)])
    score = bp * np.exp(score)
    score_list.append(score)

    return [score * 100 for score in score_list]
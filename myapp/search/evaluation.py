import math

# Evaluation metrics. Definitions follow the course slides (IRWA-04-Evaluation).
#
# `labels` is always the list of relevance labels of the results, in ranking order.
# Example: [1, 0, 1] means that the 1st and 3rd results are relevant and the 2nd is not.


def labels_in_rank_order(ranked_pids, judgments, drop_unjudged=True):
    """
    Turn a ranking of pids into the list of its relevance labels.

    :param ranked_pids: pids sorted by descending score
    :param judgments: dictionary pid -> label with the judged documents of one query
    :param drop_unjudged: True skips the documents without a label; False counts them as non-relevant
    :return: the labels in ranking order
    """
    labels = []
    for pid in ranked_pids:
        if pid in judgments:
            labels.append(judgments[pid])
        elif not drop_unjudged:
            labels.append(0)
    return labels


def count_relevant(labels):
    """Number of relevant documents in a list of labels."""
    count = 0
    for label in labels:
        if label > 0:
            count += 1
    return count


def precision_at_k(labels, k):
    """P@K: relevant documents in the top K, divided by K."""
    return count_relevant(labels[:k]) / k


def recall_at_k(labels, k, n_relevant):
    """
    R@K: relevant documents in the top K, divided by all the relevant documents of the query.

    :param n_relevant: relevant documents of the query, including the ones never retrieved
    """
    if n_relevant == 0:
        return 0.0
    return count_relevant(labels[:k]) / n_relevant


def f1_at_k(labels, k, n_relevant):
    """F1@K: harmonic mean of P@K and R@K."""
    precision = precision_at_k(labels, k)
    recall = recall_at_k(labels, k, n_relevant)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def sum_of_precisions(labels):
    """Sum of the precision at each position that holds a relevant document."""
    total = 0.0
    hits = 0
    position = 0
    for label in labels:
        position += 1
        if label > 0:
            hits += 1
            total += hits / position
    return total


def average_precision(labels, n_relevant):
    """
    AP over the whole ranking. Dividing by n_relevant means that a relevant document
    that was not retrieved counts as a precision of zero.

    :param n_relevant: relevant documents of the query, including the ones never retrieved
    """
    if n_relevant == 0:
        return 0.0
    return sum_of_precisions(labels) / n_relevant


def average_precision_at_k(labels, k, n_relevant):
    """AP@K: only the top K count. It divides by min(n_relevant, K), so a perfect top K scores 1."""
    if n_relevant == 0:
        return 0.0
    return sum_of_precisions(labels[:k]) / min(n_relevant, k)


def mean_average_precision(ap_per_query):
    """MAP: mean of the average precision of each query."""
    return sum(ap_per_query) / len(ap_per_query)


def reciprocal_rank(labels):
    """RR: 1 / position of the first relevant document (0 if there is none)."""
    position = 0
    for label in labels:
        position += 1
        if label > 0:
            return 1 / position
    return 0.0


def mean_reciprocal_rank(rr_per_query):
    """MRR: mean of the reciprocal rank of each query."""
    return sum(rr_per_query) / len(rr_per_query)


def dcg_at_k(labels, k):
    """DCG@K: sum of label / log2(position + 1) over the top K."""
    total = 0.0
    position = 0
    for label in labels[:k]:
        position += 1
        total += label / math.log2(position + 1)
    return total


def ndcg_at_k(labels, k, judged_labels):
    """
    NDCG@K: DCG@K divided by the DCG@K of the best possible ranking.

    :param judged_labels: labels of all the judged documents of the query, including the ones
                          never retrieved; sorted from high to low they are the best ranking
    """
    ideal_labels = sorted(judged_labels, reverse=True)
    ideal_dcg = dcg_at_k(ideal_labels, k)
    if ideal_dcg == 0:
        return 0.0
    return dcg_at_k(labels, k) / ideal_dcg

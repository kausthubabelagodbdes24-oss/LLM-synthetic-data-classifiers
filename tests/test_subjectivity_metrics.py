import numpy as np

from workstreams.ws3_evaluation import metrics, subjectivity


def test_diversity_metrics_on_known_vectors():
    X = np.eye(3)  # three orthogonal unit vectors: every pair at cosine distance 1
    assert metrics.remote_clique(X) == 1.0
    assert metrics.chamfer(X) == 1.0
    Y = np.vstack([X[0], X[0], X[1]])  # a duplicate pair -> nearest neighbour distance 0 for two items
    assert metrics.chamfer(Y) < 1.0
    assert metrics.faithfulness(X[:1], X) == 1.0
    assert abs(metrics.dup_rate(["A b", "a  B", "c"]) - 1 / 3) < 1e-9


def test_agreement_and_alpha():
    assert subjectivity.item_agreement(["joy", "joy", "fear"]) == ("joy", 2 / 3)
    labels = ["a", "b", "c"]
    perfect = [["a", "b", "c", "a"], ["a", "b", "c", "a"], ["a", "b", "c", "a"]]
    assert subjectivity.task_alpha(perfect, labels, "nominal") == 1.0
    noisy = [["a", "b", "c", "a"], ["b", "b", "a", "c"], ["c", "a", "c", "b"]]
    assert subjectivity.task_alpha(noisy, labels, "nominal") < 0.5


def test_survey_wins_match_titles_and_ids():
    rows = [
        {"Timestamp": "t", "Q1": "Question type (TREC-6)", "Q2": "emotion"},
        {"Timestamp": "t", "Q1": "trec", "Q2": "Fine-grained sentiment (SST-5)"},
    ]
    wins, n = subjectivity.survey_wins(rows)
    assert n == 2 and wins["trec"] == 2 and wins["emotion"] == 1 and wins["sst5"] == 1

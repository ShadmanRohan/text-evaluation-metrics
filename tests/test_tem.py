import unittest
from text_evaluation_metrics.tem import bleu_score, rouge_score

class TestTextEvaluationMetrics(unittest.TestCase):

    def test_bleu_score(self):
        candidate = "The quick brown dog jumped over the lazy fox."
        references = [
            "The quick brown dog jumps over the lazy fox.",
            "The fast brown canine leaps over the lazy vulpine."
        ]
        score = bleu_score(candidate, references)
        self.assertGreater(score, 0)
        self.assertLess(score, 1)

    def test_rouge_score(self):
        candidate = "The quick brown dog jumped over the lazy fox."
        references = [
            "The quick brown dog jumps over the lazy fox.",
            "The fast brown canine leaps over the lazy vulpine."
        ]
        score = rouge_score(candidate, references, n=2)
        self.assertGreater(score, 0)
        self.assertLess(score, 1)

if __name__ == "__main__":
    unittest.main()

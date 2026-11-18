"""PhraseMerger must merge tokens exactly like gensim FrozenPhrases."""
import random
import unittest

from lexical.encoders.word_vectors import PhraseMerger


class PhraseMerging(unittest.TestCase):
    def test_hand_made(self):
        merger = PhraseMerger({"machine_learning": 50.0, "board_of_directors": 30.0, "low_score": 1.0},
                              threshold=10, connector_words={"of", "the"})
        self.assertEqual(merger[["machine", "learning", "model"]], ["machine_learning", "model"])
        self.assertEqual(merger[["board", "of", "directors"]], ["board_of_directors"])
        self.assertEqual(merger[["the", "board", "of", "the"]], ["the", "board", "of", "the"])
        self.assertEqual(merger[["low", "score"]], ["low", "score"])          # below threshold
        self.assertEqual(merger[[]], [])

    def test_matches_gensim(self):
        try:
            from gensim.models.phrases import ENGLISH_CONNECTOR_WORDS, Phrases
        except ImportError:
            self.skipTest("gensim not installed (only needed for this cross-check)")
        rng = random.Random(0)
        words = ["data", "base", "machine", "learning", "board", "of", "directors", "the", "chief",
                 "executive", "officer", "and", "neural", "network"]
        corpus = [["machine", "learning"], ["board", "of", "directors"], ["chief", "executive", "officer"],
                  ["neural", "network"], ["data", "base"]] * 20
        corpus += [[rng.choice(words) for _ in range(6)] for _ in range(200)]
        bigram = Phrases(corpus, min_count=2, threshold=1, connector_words=ENGLISH_CONNECTOR_WORDS).freeze()
        trigram = Phrases([bigram[s] for s in corpus], min_count=2, threshold=1,
                          connector_words=ENGLISH_CONNECTOR_WORDS).freeze()
        ours = [PhraseMerger(p.phrasegrams, p.threshold, p.delimiter, p.connector_words) for p in (bigram, trigram)]
        for _ in range(2000):
            tokens = [rng.choice(words) for _ in range(rng.randint(1, 7))]
            self.assertEqual(ours[1][ours[0][tokens]], trigram[bigram[tokens]])


if __name__ == "__main__":
    unittest.main()

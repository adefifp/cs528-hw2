import unittest
from main import run_pagerank, compute_closeness_centrality

class TestGraphAlgorithms(unittest.TestCase):
    def test_pagerank_directed_cycle(self):
        n = 3
        out_edges = [[1], [2], [0]]
        in_edges = [[2], [0], [1]]
        
        pr = run_pagerank(n, out_edges, in_edges, damping=0.85, threshold=1e-5)
        for score in pr:
            self.assertAlmostEqual(score, 1.0 / 3.0, places=4)

    def test_closeness_centrality_star_graph(self):
        n = 4
        out_edges = [[1, 2, 3], [], [], []]
        
        best_node, score = compute_closeness_centrality(n, out_edges)
        self.assertEqual(best_node, 0)
        self.assertGreater(score, 0.0)

if __name__ == "__main__":
    unittest.main()
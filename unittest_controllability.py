import numpy as np
import networkx as nx
import unittest
import pandas as pd
from controllability import *

class TestControllabilityModules(unittest.TestCase):
    def setUp(self):
        self.fn_graph = '/Users/skyeong/pythonwork/controllability/data/testgraph.edgelist'
        self.fn_avg = '/Users/skyeong/pythonwork/controllability/data/averagectl.txt'
        self.fn_modal = '/Users/skyeong/pythonwork/controllability/data/modalctl.txt'
        self.load_graph()

    def load_graph(self):
        # Load test data
        G=nx.read_edgelist(self.fn_graph)
        nodelist = [str(i+1) for i in range(82)]
        self.A=nx.to_numpy_matrix(G,nodelist=nodelist)   

    def test_average_controllability(self):
        data = pd.read_csv(self.fn_avg)
        actual_value = np.array(data['value'])
        computed_value = ave_control(self.A)
        
        # Make sure the two arrays have the same length
        self.assertAlmostEqual(np.mean(actual_value), np.mean(computed_value), places=6)
        # self.assertAlmostEqual(actual_value, computed_value, places=2)

    def test_modal_controllability(self):
        computed_value = modal_control(self.A)
        data = pd.read_csv(self.fn_modal)
        actual_value = np.array(data['value'])

        # Make sure the two arrays have the same length
        self.assertAlmostEqual(np.mean(actual_value), np.mean(computed_value), places=6)
        # self.assertAlmostEqual(actual_value, computed_value, places=2)
    
    # def test_structural_controllability(self):
        
if __name__=="__main__":
    unittest.main()

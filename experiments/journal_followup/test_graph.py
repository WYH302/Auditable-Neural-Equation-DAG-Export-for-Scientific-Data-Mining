import unittest
import numpy as np
import torch

from graph import cse,evaluate,measured,token_count
from models import MLP,EMLKAN,RBFKAN


class ScalarExportTests(unittest.TestCase):
    def test_numeric_literal_precision_does_not_change_token_count(self):
        self.assertEqual(token_count('y = 1.2e-10*x1'),token_count('y = 0.123456789*x1'))

    def test_cse_definitions_are_dependency_ordered(self):
        raw=['a = x1 + 1','b = exp(a) + exp(a)','y = b + exp(a)']
        opt=cse(raw)
        x=np.linspace(-2,2,30).reshape(-1,1)
        np.testing.assert_allclose(evaluate(raw,x),evaluate(opt,x),rtol=1e-13)

    def test_all_model_exports_and_optimized_variants_match(self):
        torch.set_num_threads(1)
        torch.manual_seed(5)
        rng=np.random.default_rng(5)
        for dim in (1,2):
            x=rng.uniform(-2,3,(50,dim))
            scales=dict(x_mean=[0.7]*dim,x_std=[1.2]*dim,y_mean=-0.3,y_std=1.7)
            for method,model in [('mlp',MLP(dim,4,2)),('eml_kan',EMLKAN(dim,4,2)),('kan',RBFKAN(dim,4))]:
                model=model.double()
                xn=(x-np.array(scales['x_mean']))/np.array(scales['x_std'])
                with torch.no_grad():pred=model(torch.from_numpy(xn)).numpy().reshape(-1)*1.7-0.3
                stats,graphs=measured(model,method,scales,x,pred)
                self.assertTrue(all(v<1e-9 for v in stats['max_abs_errors'].values()))

    def test_log_input_offset_is_serialized(self):
        model=EMLKAN(1,2,1).double()
        x=np.linspace(-1,1,40).reshape(-1,1)
        scales=dict(x_mean=[0.0],x_std=[1.0],y_mean=0.0,y_std=1.0)
        with torch.no_grad():pred=model(torch.from_numpy(x)).numpy().reshape(-1)
        _,graphs=measured(model,'eml_kan',scales,x,pred)
        self.assertTrue(any('1e-6' in line for line in graphs['raw']))


if __name__=='__main__':
    unittest.main()

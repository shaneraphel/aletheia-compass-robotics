import importlib.util, json, sys, traceback
import numpy as np
path = sys.argv[1]
try:
    spec = importlib.util.spec_from_file_location("aletheia_live_candidate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    def reference(points):
        x = np.asarray(points, dtype=np.float64)
        xc = x-x.mean(axis=0)
        gram = xc@xc.T
        norms = np.diag(gram)
        distances = np.maximum(norms[:,None]+norms[None,:]-2*gram, 0.0)
        centering = np.eye(len(x))-np.ones((len(x),len(x)))/len(x)
        return gram, -0.5*centering@distances@centering
    rng = np.random.default_rng(20261114)
    cases = 0
    maximum = 0.0
    for n in (1, 2, 7, 31, 130):
        for block_size in (1, 5, 256):
            points = rng.normal(size=(n, 3))
            result = module.blocked_gram_dual_route(points, block_size)
            assert set(result) == {"gram_dual_route_max_abs", "blocks", "gram_dual_route_finite"}
            assert result["blocks"] == (n+block_size-1)//block_size
            assert result["gram_dual_route_finite"] is True
            gram, dual = reference(points)
            blocked_error = result["gram_dual_route_max_abs"]
            dense_error = float(np.max(np.abs(gram-dual)))
            assert blocked_error < 2e-11, (n, block_size, blocked_error, dense_error)
            maximum = max(maximum, blocked_error, dense_error)
            cases += 1
    repeated = np.repeat(np.array([[1.0, -2.0, 0.5]]), 17, axis=0)
    assert module.blocked_gram_dual_route(repeated, 4)["gram_dual_route_max_abs"] == 0.0
    for invalid in (np.empty((0,3)), np.ones((3,2)), np.full((3,3), np.nan)):
        try:
            module.blocked_gram_dual_route(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid point cloud was accepted")
    for invalid_block in (0, -1, 1.5):
        try:
            module.blocked_gram_dual_route(np.ones((3,3)), invalid_block)
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError("invalid block size was accepted")
    report = {"passed": True, "random_block_cases": cases+1,
              "maximum_dense_or_blocked_identity_error": maximum,
              "invalid_input_contracts_passed": True}
except Exception:
    report = {"passed": False, "traceback": traceback.format_exc()}
print(json.dumps(report, indent=2))
sys.exit(0 if report["passed"] else 1)

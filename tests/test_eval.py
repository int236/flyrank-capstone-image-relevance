from app.eval import run_eval

def test_eval_precision_meets_bar():
    result = run_eval()
    assert result["total"] == 5
    assert result["top1_precision"] >= 0.8, result["details"]

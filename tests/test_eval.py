import math
from trainer.eval_metrics import (
    prepare_tool_eval_examples,
    eval_tool_calls,
    _parse_predicted_tool_calls,
    _f1_precision_recall,
)


def test__parse_predicted_tool_calls_roundtrip():
    text = (
        "<tool_call>"
        '{"id":"call_1","type":"function",'
        '"function":{"name":"lookup_inventory","arguments":"{\\"vin\\": \\"123\\"}"}}'
        "</tool_call>"
    )
    calls, fails = _parse_predicted_tool_calls(text)
    assert fails == 0
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "lookup_inventory"
    assert '"vin"' in calls[0]["function"]["arguments"]


def test__f1_precision_recall_exact_match():
    prec, rec, f1 = _f1_precision_recall(
        ["lookup_inventory", "schedule_test_drive"],
        ["lookup_inventory", "schedule_test_drive"],
    )
    assert prec == 1.0
    assert rec == 1.0
    assert f1 == 1.0


def test_tool_eval_metrics_vllm_smoke(
    tiny_vllm_model_dir,
    raw_sessions_fixture,
    global_tools_fixture,
):
    """
    Real vLLM call.
    We only assert shape/bounds because the small model may not follow the format exactly.
    """
    model_path, tok = tiny_vllm_model_dir

    examples = prepare_tool_eval_examples(raw_sessions_fixture)

    metrics = eval_tool_calls(
        model_path=model_path,
        tokenizer=tok,
        examples=examples,
        global_tools=global_tools_fixture,
        max_model_len=4096,
        max_new_tokens=128,
        temperature=0.0,
        tensor_parallel_size=1,
    )

    expected_keys = {
        "tool_call_precision",
        "tool_call_recall",
        "tool_call_name_f1",
        "arg_exact_match_rate",
        "arguments_parse_success_rate",
        "no_tool_hallucination_rate",
        "num_tool_turns",
        "num_no_tool_turns",
    }
    assert expected_keys.issubset(metrics.keys())

    for k in [
        "tool_call_precision",
        "tool_call_recall",
        "tool_call_name_f1",
        "arg_exact_match_rate",
        "arguments_parse_success_rate",
        "no_tool_hallucination_rate",
    ]:
        v = metrics[k]
        assert isinstance(v, float)
        assert math.isfinite(v)
        assert 0.0 <= v <= 1.0

    assert metrics["num_tool_turns"] >= 0
    assert metrics["num_no_tool_turns"] >= 0

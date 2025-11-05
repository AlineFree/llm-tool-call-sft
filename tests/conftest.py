import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer


@pytest.fixture(scope="session")
def tiny_vllm_model_dir(tmp_path_factory):
    """
    Load a small HF model once, save it to a temp dir, and return (path, tokenizer).
    vLLM will read from this local dir.
    """
    model_name = "viktoroo/SmolLM2-360M-Tools"
    out_dir = tmp_path_factory.mktemp("hf_model")

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id

    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)

    # return path as str so vLLM can consume it
    return str(out_dir), tok


@pytest.fixture
def global_tools_fixture():
    return [
        {
            "type": "function",
            "function": {
                "name": "lookup_inventory",
                "description": "Return availability details for a specific vehicle VIN.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "vin": {
                            "type": "string",
                            "description": "Vehicle VIN the shopper asked about."
                        },
                    },
                    "required": ["vin"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "schedule_test_drive",
                "description": "Book a test drive appointment for a VIN at a specific datetime.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "vin": {
                            "type": "string",
                            "description": "Vehicle VIN to test drive."
                        },
                        "datetime": {
                            "type": "string",
                            "description": "Requested start time in ISO 8601."
                        },
                    },
                    "required": ["vin", "datetime"],
                },
            },
        },
    ]


@pytest.fixture
def raw_sessions_fixture(global_tools_fixture):
    # unchanged from your version
    tool_doc_lines = [
        "You are CarSalesBot for a dealership.",
        "You have function calling capabilities. You can call these functions:",
    ]
    for t in global_tools_fixture:
        fn = t["function"]["name"]
        desc = t["function"]["description"]
        tool_doc_lines.append(f"- {fn}: {desc}")
    tool_doc_lines.extend([
        "",
        "When the shopper asks about a specific vehicle by VIN or asks if a car is in stock:",
        "1. You MUST respond with EXACTLY one <tool_call>...</tool_call> block.",
        "2. Inside that block output valid JSON:",
        '   {"id":"call_1","type":"function","function":{"name":"lookup_inventory","arguments":"{\\"vin\\": \\"<VIN>\\"}"}}',
        "3. Do NOT add natural language before or after the block.",
        "",
        "After I send you the tool result from lookup_inventory, then respond in natural language.",
        "",
        "When the shopper is just greeting or saying hi:",
        "Respond conversationally as plain text.",
        "Do NOT call any tool in that case.",
    ])
    system_prompt = "\n".join(tool_doc_lines)

    session_a = {
        "session_id": "sess_a",
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "Shopper: I'm looking at a used Honda Civic. The VIN is 123. "
                    "Follow the policy. Return ONLY the tool call JSON in a <tool_call> block "
                    "so the dealership system can check availability."
                ),
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "lookup_inventory",
                            "arguments": '{"vin": "123"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": '{"vin":"123","available":true,"price_usd":15999}',
            },
            {
                "role": "assistant",
                "content": "Yes. The Civic with VIN 123 is in stock and listed at $15,999.",
            },
        ],
    }

    session_b = {
        "session_id": "sess_b",
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "Shopper: Just say hi to me. Do NOT call any tool. "
                    "You are only greeting me."
                ),
            },
            {
                "role": "assistant",
                "content": "Hello. How can I help you with a vehicle today?",
            },
        ],
    }

    return [session_a, session_b]

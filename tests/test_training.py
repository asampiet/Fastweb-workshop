"""T-401: Unit tests for fine-tuning dataset and training configuration."""
import json
import pytest

TRAIN_PATH = "data/training_data.json"
TEST_PATH = "data/test_data.json"

@pytest.fixture(scope="module")
def train_data():
    with open(TRAIN_PATH) as f:
        return json.load(f)

@pytest.fixture(scope="module")
def test_data():
    with open(TEST_PATH) as f:
        return json.load(f)

def test_training_dataset_count(train_data):
    assert len(train_data) == 1300, f"Expected 1300 examples, got {len(train_data)}"

def test_test_dataset_count(test_data):
    assert len(test_data) == 1000, f"Expected 1000 examples, got {len(test_data)}"

def test_training_data_has_text_field(train_data):
    for i, ex in enumerate(train_data):
        assert "text" in ex, f"Example {i} missing 'text' field"

def test_chatml_format(train_data):
    """Every example must have system, user, and assistant turns."""
    for i, ex in enumerate(train_data):
        text = ex["text"]
        assert "<|im_start|>system" in text, f"Example {i}: missing system turn"
        assert "<|im_start|>user" in text, f"Example {i}: missing user turn"
        assert "<|im_start|>assistant" in text, f"Example {i}: missing assistant turn"

def test_system_prompt_consistent(train_data):
    """All examples must use the same system prompt."""
    expected = "You are a 3GPP Protocol Expert specializing in Root Cause Analysis."
    for i, ex in enumerate(train_data):
        assert expected in ex["text"], f"Example {i}: system prompt mismatch"

def test_assistant_output_is_json(train_data):
    """Every assistant turn must contain valid JSON with 'events' key."""
    for i, ex in enumerate(train_data):
        text = ex["text"]
        marker = "<|im_start|>assistant\n"
        idx = text.find(marker)
        assert idx != -1, f"Example {i}: no assistant marker"
        start = idx + len(marker)
        end = text.find("<|im_end|>", start)
        assistant_text = text[start:end].strip() if end != -1 else text[start:].strip()
        data = json.loads(assistant_text)
        assert "events" in data, f"Example {i}: assistant JSON missing 'events'"

def test_has_positive_examples(train_data):
    """Dataset must contain examples with non-empty events."""
    positives = 0
    for ex in train_data:
        text = ex["text"]
        marker = "<|im_start|>assistant\n"
        start = text.find(marker) + len(marker)
        end = text.find("<|im_end|>", start)
        data = json.loads(text[start:end].strip())
        if len(data["events"]) > 0:
            positives += 1
    assert positives >= 1000, f"Expected ≥1000 positive examples, got {positives}"

def test_has_negative_examples(train_data):
    """Dataset must contain examples with empty events (normal logs)."""
    negatives = 0
    for ex in train_data:
        text = ex["text"]
        marker = "<|im_start|>assistant\n"
        start = text.find(marker) + len(marker)
        end = text.find("<|im_end|>", start)
        data = json.loads(text[start:end].strip())
        if len(data["events"]) == 0:
            negatives += 1
    assert negatives >= 100, f"Expected ≥100 negative examples, got {negatives}"

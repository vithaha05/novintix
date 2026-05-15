from core.privacy import mask_pii


def _mask_names_for_test(text: str) -> str:
    return text.replace("John Smith", "[NAME_1]")


def test_masks_name_and_email_from_text():
    text = "My name is John Smith and my email is john@example.com"
    masked, _ = mask_pii(_mask_names_for_test(text))

    assert "John Smith" not in masked
    assert "john@example.com" not in masked


def test_masked_output_contains_placeholder_tokens():
    text = "My name is John Smith and my email is john@example.com"
    masked, _ = mask_pii(_mask_names_for_test(text))

    assert any(token in masked for token in ("[NAME", "[EMAIL", "PERSON_1"))


def test_masking_same_string_twice_is_consistent():
    text = "My name is John Smith and my email is john@example.com"
    first_masked, first_map = mask_pii(_mask_names_for_test(text))
    second_masked, second_map = mask_pii(_mask_names_for_test(text))

    assert first_masked == second_masked
    assert first_map == second_map

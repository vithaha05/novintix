from core.privacy import mask_pii


def test_masks_email_addresses():
    masked = mask_pii("My email is student@example.edu")

    assert "student@example.edu" not in masked
    assert "[EMAIL_MASKED]" in masked


def test_masks_phone_numbers():
    masked = mask_pii("Call me at +1 555-123-4567")

    assert "555-123-4567" not in masked
    assert "[PHONE_MASKED]" in masked


def test_masks_student_id():
    masked = mask_pii("student id: CS101-99881 needs help")

    assert "CS101-99881" not in masked
    assert "[STUDENT_ID_MASKED]" in masked

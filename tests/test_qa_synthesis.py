from run_rag_demo import _is_structural_noise, build_synthesis_messages, extract_highest_scoring_sentence_window


def test_sentence_window_keeps_answer_sentence_and_neighbours():
    text = "The story begins quietly. The river was brown after the rain. Everyone crossed it safely."
    _, window, score = extract_highest_scoring_sentence_window(text, "What color was the river?")

    assert "river was brown" in window
    assert "Everyone crossed" in window
    assert score > 0


def test_structural_document_furniture_is_detected():
    assert _is_structural_noise("PROJECT GUTENBERG LICENSE AND TABLE OF CONTENTS")
    assert not _is_structural_noise("The river in the story is brown.")


def test_synthesis_messages_are_model_agnostic():
    messages = build_synthesis_messages("The river is brown.", "What color is it?")

    assert messages[0]["role"] == "system"
    assert "The Man Elephant" not in messages[1]["content"]
    assert "Story Context:\nThe river is brown." in messages[1]["content"]
    assert "Question: What color is it?" in messages[1]["content"]

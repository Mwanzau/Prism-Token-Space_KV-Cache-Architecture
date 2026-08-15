from prism_protocol.src.token_space.library import LibraryDocumentParser, add_document_alias, empty_library, route_section


def test_parser_hashes_sections_and_detects_markdown_and_book_headings():
    sections = LibraryDocumentParser().parse_document(
        "Preamble text.\n# The Man Elephant\nBrown river.\nTHE FLYING LION\nSky story.",
        "tales.txt",
    )

    assert [section["section_title"] for section in sections] == ["Preamble", "The Man Elephant", "The Flying Lion"]
    assert len({section["section_hash"] for section in sections}) == 3
    assert all(section["section_id"].startswith("SECTION_") for section in sections)


def test_router_locks_second_stage_to_the_winning_section():
    parser = LibraryDocumentParser()
    elephant, lion = parser.parse_document("# Man Elephant\nRiver brown.\n# Flying Lion\nSky blue.", "tales.txt")[1:]
    library = empty_library(1)
    for chunk_id, section in enumerate((elephant, lion)):
        add_document_alias(library, "tales.txt", section)
        library["sections"][section["section_hash"]]["chunk_ids"].append(chunk_id)
        library["chunks"][str(chunk_id)] = {"section_hash": section["section_hash"]}

    digest, chunk_ids, _ = route_section("What happens to the man elephant?", {0: 1.0, 1: 1.5}, library)

    assert digest == elephant["section_hash"]
    assert chunk_ids == {0}

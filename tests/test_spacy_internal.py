from math import isclose

import pytest
import spacy

from classy_classification.examples.data import training_data, validation_data


@pytest.fixture
def spacy_internal():
    nlp = spacy.load("en_core_web_md")
    nlp.add_pipe(
        "text_categorizer",
        config={
            "data": training_data,
            "model": "spacy",
            "include_sent": True,
        },
    )
    return nlp


def test_spacy_internal(spacy_internal):
    doc = spacy_internal(validation_data[0])
    assert isclose(sum(doc._.cats.values()), 1)
    for sent in doc.sents:
        assert isclose(sum(sent._.cats.values()), 1)

    docs = spacy_internal.pipe(validation_data)
    for doc in docs:
        assert isclose(sum(doc._.cats.values()), 1)
        for sent in doc.sents:
            assert isclose(sum(sent._.cats.values()), 1)

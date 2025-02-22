import pytest
import spacy

from classy_classification.examples.data import training_data, validation_data


@pytest.fixture
def spacy_external_multi_label():
    nlp = spacy.blank("en")
    nlp.add_pipe(
        "text_categorizer",
        config={
            "data": training_data,
            "include_sent": True,
        },
    )
    return nlp


def test_spacy_external_multi_label(spacy_external_multi_label):
    doc = spacy_external_multi_label(validation_data[0])
    assert doc._.cats
    for sent in doc.sents:
        assert sent._.cats

    docs = spacy_external_multi_label.pipe(validation_data)
    for doc in docs:
        assert doc._.cats
        for sent in doc.sents:
            assert sent._.cats

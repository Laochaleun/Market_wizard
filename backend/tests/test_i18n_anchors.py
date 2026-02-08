"""Tests for SSR anchor variants in i18n."""

from app.i18n import (
    ANCHOR_SETS_VARIANTS,
    DEFAULT_ANCHOR_VARIANT,
    Language,
    get_anchor_sets,
    get_anchor_variants,
)


def test_default_anchor_variant_is_registered():
    assert DEFAULT_ANCHOR_VARIANT in get_anchor_variants()


def test_all_anchor_variants_are_bilingual_and_complete():
    for variant in get_anchor_variants():
        assert variant in ANCHOR_SETS_VARIANTS
        for language in (Language.PL, Language.EN):
            anchor_sets = get_anchor_sets(language, variant=variant)
            assert len(anchor_sets) == 6
            for anchor_set in anchor_sets:
                assert set(anchor_set.keys()) == {1, 2, 3, 4, 5}
                for score in range(1, 6):
                    text = anchor_set[score]
                    assert isinstance(text, str)
                    assert text.strip()


def test_v4_anchor_variant_exists_for_polish_and_english():
    assert len(get_anchor_sets(Language.PL, variant="paper_general_v4")) == 6
    assert len(get_anchor_sets(Language.EN, variant="paper_general_v4")) == 6


def test_v41_anchor_variant_exists_for_polish_and_english():
    assert len(get_anchor_sets(Language.PL, variant="paper_general_v4.1")) == 6
    assert len(get_anchor_sets(Language.EN, variant="paper_general_v4.1")) == 6

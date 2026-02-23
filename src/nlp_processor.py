import inspect
from collections import namedtuple
from functools import lru_cache


def _ensure_inspect_compatibility():
    # pymorphy2 expects inspect.getargspec, removed in Python 3.11+.
    if hasattr(inspect, "getargspec"):
        return

    arg_spec = namedtuple("ArgSpec", ["args", "varargs", "keywords", "defaults"])

    def _getargspec(func):
        full = inspect.getfullargspec(func)
        return arg_spec(full.args, full.varargs, full.varkw, full.defaults)

    inspect.getargspec = _getargspec


_ensure_inspect_compatibility()

from natasha import (
    DatesExtractor,
    Doc,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    NewsNERTagger,
    NewsSyntaxParser,
    Segmenter,
)


def _unique(values):
    seen = set()
    result = []
    for value in values:
        normalized = value.strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(normalized)
    return result


class NLPProcessor:
    def __init__(self):
        self.backend = "natasha"
        embedding = NewsEmbedding()
        morph_vocab = MorphVocab()
        self._pipeline = {
            "Doc": Doc,
            "segmenter": Segmenter(),
            "morph_tagger": NewsMorphTagger(embedding),
            "syntax_parser": NewsSyntaxParser(embedding),
            "ner_tagger": NewsNERTagger(embedding),
            "morph_vocab": morph_vocab,
            "dates_extractor": DatesExtractor(morph_vocab),
        }

    def analyze(self, text):
        text = text if isinstance(text, str) else ""
        doc = self._pipeline["Doc"](text)
        doc.segment(self._pipeline["segmenter"])
        doc.tag_morph(self._pipeline["morph_tagger"])
        doc.parse_syntax(self._pipeline["syntax_parser"])
        doc.tag_ner(self._pipeline["ner_tagger"])

        people = []
        locations = []
        organizations = []
        lemmas = []

        morph_vocab = self._pipeline["morph_vocab"]
        for token in doc.tokens:
            token.lemmatize(morph_vocab)
            lemma = (token.lemma or token.text).lower()
            if lemma.isalpha() and len(lemma) > 1:
                lemmas.append(lemma)

        for span in doc.spans:
            span.normalize(morph_vocab)
            value = getattr(span, "normal", None) or span.text
            if span.type == "PER":
                people.append(value)
            elif span.type == "LOC":
                locations.append(value)
            elif span.type == "ORG":
                organizations.append(value)

        dates = []
        for match in self._pipeline["dates_extractor"](text):
            start, stop = match.start, match.stop
            dates.append(text[start:stop])

        return {
            "backend": self.backend,
            "people": _unique(people),
            "locations": _unique(locations),
            "dates": _unique(dates),
            "organizations": _unique(organizations),
            "lemmas": _unique(lemmas),
        }


@lru_cache(maxsize=1)
def get_nlp_processor():
    return NLPProcessor()


def analyze_text(text):
    return get_nlp_processor().analyze(text)


def build_nlp_summary(analysis):
    people = analysis.get("people", [])
    locations = analysis.get("locations", [])
    dates = analysis.get("dates", [])
    organizations = analysis.get("organizations", [])
    lemmas = analysis.get("lemmas", [])

    lines = []
    if people:
        lines.append(f"Имена: {', '.join(people)}")
    if locations:
        lines.append(f"Локации: {', '.join(locations)}")
    if dates:
        lines.append(f"Даты: {', '.join(dates)}")
    if organizations:
        lines.append(f"Организации: {', '.join(organizations)}")
    if lemmas:
        lines.append(f"Леммы: {', '.join(lemmas[:12])}")

    if not lines:
        return ""

    return "Распознанные сущности и леммы:\n" + "\n".join(lines)

import warnings
from functools import lru_cache
from importlib.metadata import entry_points

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)


def _patch_pymorphy2_entry_points():
    try:
        import pkg_resources  # type: ignore
    except ModuleNotFoundError:
        pkg_resources = None

    if pkg_resources is not None:
        return

    import pymorphy2.analyzer as pymorphy2_analyzer

    def _iter_entry_points(*args, **kwargs):
        group = kwargs.get("group")
        name = kwargs.get("name")
        if args:
            group = args[0]
        if len(args) > 1:
            name = args[1]

        discovered = entry_points()
        if hasattr(discovered, "select"):
            filters = {}
            if group is not None:
                filters["group"] = group
            if name is not None:
                filters["name"] = name
            return tuple(discovered.select(**filters))

        candidates = tuple(discovered.get(group, ())) if group is not None else tuple(discovered)
        if name is not None:
            candidates = tuple(item for item in candidates if item.name == name)
        return candidates

    pymorphy2_analyzer._iter_entry_points = _iter_entry_points


_patch_pymorphy2_entry_points()

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


class TextLemmatizer:
    def __init__(self):
        embedding = NewsEmbedding()
        self._doc_type = Doc
        self._segmenter = Segmenter()
        self._morph_tagger = NewsMorphTagger(embedding)
        self._morph_vocab = MorphVocab()

    def lemmatize(self, text):
        text = text if isinstance(text, str) else ""
        if not text.strip():
            return []

        doc = self._doc_type(text)
        doc.segment(self._segmenter)
        doc.tag_morph(self._morph_tagger)

        lemmas = []
        for token in doc.tokens:
            token.lemmatize(self._morph_vocab)
            lemma = (token.lemma or token.text).strip().lower()
            if lemma:
                lemmas.append(lemma)
        return lemmas


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
def get_text_lemmatizer():
    return TextLemmatizer()


@lru_cache(maxsize=65536)
def lemmatize_text(text):
    return tuple(get_text_lemmatizer().lemmatize(text))


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

import re
import nltk
from nltk.corpus import stopwords
import spacy

nltk.download('stopwords')
en_stopwords = set(stopwords.words('english'))

nlp = spacy.load("en_core_web_md")  

def preprocess_keep_or_exclude_pos(
    text: str,
    keep_pos: set[str]        = None,
    excluded_pos: set[str]    = None,
    stop_words: set[str]      = None,
    custom_excluded: set[str] = None,
    excluded_lemmas: set[str] = None
) -> list[str]:
    """
    Preprocess text, either keeping only `keep_pos` tags, or—if keep_pos is None—
    dropping any token whose POS is in `excluded_pos`.
    """
    # 1) POS filters
    if keep_pos is not None:
        keep_pos        = set(keep_pos)
        use_keep_filter = True
    else:
        excluded_pos    = set(excluded_pos or {'PRON','DET','ADP','CCONJ','SCONJ','PART','INTJ'})
        use_keep_filter = False

    # 2) stop‐lists
    stop_words      = set(w.lower() for w in (stop_words or en_stopwords))
    custom_excluded = set(custom_excluded or [])
    excluded_lemmas = set(excluded_lemmas or [])

    # 3) normalize & strip
    t = text.lower()
    t = re.sub(r'[^a-záéíóúüñ\s]', " ", t)
    t = re.sub(r"\s+", " ", t).strip()

    # 4) spaCy tokenize + lemmatize
    doc = nlp(t)
    out = []
    for tok in doc:
        lemma = tok.lemma_.lower()
        pos_ok = (use_keep_filter and tok.pos_ in keep_pos) or \
                 (not use_keep_filter and tok.pos_ not in excluded_pos)

        if (
            pos_ok
            and tok.is_alpha
            and lemma not in stop_words
            and lemma not in custom_excluded
            and lemma not in excluded_lemmas
        ):
            out.append(lemma)

    return out
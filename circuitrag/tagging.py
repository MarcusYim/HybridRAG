"""Paper-level tags: a curated regex vocabulary plus TF-IDF keyphrases."""
from __future__ import annotations

import re

import numpy as np

from .text import STOPWORDS


TAG_VOCAB: dict[str, list[tuple[str, str]]] = {
    "organism": [
        ("E. coli", r"\bE\.\s?coli\b|\bEscherichia coli\b"),
        ("yeast", r"\byeast\b|\bS\.\s?cerevisiae\b|\bSaccharomyces\b"),
        ("mammalian", r"\bmammalian\b|\bHEK\s?-?293\w*|\bCHO cells?\b|\bhuman cells?\b"),
        ("B. subtilis", r"\bB\.\s?subtilis\b|\bBacillus subtilis\b"),
        ("cell-free", r"\bcell[- ]free\b|\bTX-?TL\b|\bPURE system\b"),
        ("plant", r"\bArabidopsis\b|\bNicotiana\b|\bplant cells?\b"),
    ],
    "function": [
        ("toggle switch / bistability", r"\btoggle switch(?:es)?\b|\bbistab\w*"),
        ("oscillator", r"\boscillat\w*|\brepressilator\b"),
        ("logic gate", r"\blogic gates?\b|\b(?:AND|OR|NOR|NAND|XOR|NOT)[- ]gates?\b|\bboolean logic\b"),
        ("feedback control", r"\b(?:negative|positive|integral) feedback\b|\bfeedback control\w*|\bantithetic\b"),
        ("feed-forward loop", r"\bfeed-?forward loops?\b|\bIFFLs?\b"),
        ("memory", r"\b(?:cellular|genetic|molecular|transcriptional) memory\b|\brecording device"),
        ("biosensor", r"\bbiosensors?\b|\bsensing circuits?\b"),
        ("counter", r"\b(?:genetic|molecular) counters?\b"),
        ("pulse / band-pass", r"\bpulse generat\w*|\bedge detect\w*|\bband-?pass\b"),
        ("quorum sensing", r"\bquorum[- ]sensing\b"),
        ("multicellular / consortia", r"\bconsorti(?:a|um)\b|\bco-?cultures?\b|\bmulticellular\b"),
    ],
    "mechanism": [
        ("transcriptional", r"\btranscription(?:al)? (?:factors?|repress\w*|activat\w*|regulat\w*)|\brepressors?\b"),
        ("CRISPR", r"\bCRISPR\w*|\bdCas9\b|\bCas9\b|\bsgRNAs?\b"),
        ("recombinase", r"\brecombinases?\b|\bintegrases?\b|\bBxb1\b"),
        ("RNA-based", r"\briboregulators?\b|\btoehold switch\w*|\bsRNAs?\b|\briboswitch\w*|\bsmall RNAs?\b"),
        ("post-translational", r"\bphosphorylation\b|\bproteases?\b|\bdegrons?\b|\bpost-?translational\b"),
        ("optogenetic", r"\boptogenetic\w*|\blight[- ]induc\w*"),
        ("resource competition / burden", r"\bresource (?:competition|sharing|loading)\b|\bretroactivity\b|\b(?:metabolic|cellular|expression) burden\b"),
    ],
    "method": [
        ("ODE model", r"\bordinary differential equations?\b|\bODEs?\b|\bmass[- ]action\b"),
        ("stochastic model", r"\bstochastic\b|\bGillespie\b|\bchemical master equation\b"),
        ("Hill function", r"\bHill (?:functions?|coefficients?|equations?)\b"),
        ("flow cytometry", r"\bflow cytometry\b|\bFACS\b"),
        ("microfluidics", r"\bmicrofluidic\w*|\bmother machine\b"),
        ("design automation", r"\bCello\b|\bdesign automation\b|\bcomputer-aided design\b"),
        ("parameter inference", r"\bparameter (?:inference|estimation|identification)\b"),
        ("control theory", r"\bcontrol theor\w*|\bPID\b|\brobust (?:control|perfect adaptation)\b"),
    ],
    # Part names are case-sensitive on purpose: cI is not CI, AraC is not "arac".
    "part": [
        (name, pat)
        for name, pat in [
            ("TetR", r"\bTetR\b"), ("LacI", r"\bLacI\b"), ("cI (lambda)", r"\b(?:λ|lambda )?cI\b"),
            ("AraC", r"\bAraC\b"), ("LuxR", r"\bLuxR\b"), ("LuxI", r"\bLuxI\b"), ("PhlF", r"\bPhlF\b"),
            ("SrpR", r"\bSrpR\b"), ("AmtR", r"\bAmtR\b"), ("BetI", r"\bBetI\b"), ("QacR", r"\bQacR\b"),
            ("T7 RNAP", r"\bT7 RNA ?(?:P|polymerase)\b"), ("Ptet", r"\b[pP]\s?[Tt]et[O]?\b"),
            ("Plac", r"\b[pP]\s?lac\b|\bPlacO\b"), ("pBAD", r"\b[pP]BAD\b"), ("Plux", r"\b[pP]\s?lux\b"),
            ("aTc", r"\baTc\b|\banhydrotetracycline\b"), ("IPTG", r"\bIPTG\b"), ("AHL", r"\bAHLs?\b|\bHSL\b"),
            ("arabinose", r"\barabinose\b"), ("GFP", r"\b(?:sf|e)?GFP\b"), ("mCherry", r"\bmCherry\b"),
            ("RFP", r"\bm?RFP\b"), ("YFP", r"\b(?:e)?YFP\b"), ("ssrA / LVA tag", r"\bssrA\b|\bLVA\b|\bLAA\b"),
            ("ECF sigma", r"\bECF sigma\b|\bECF σ"),
        ]
    ],
}
_COMPILED_VOCAB = {
    cat: [(label, re.compile(p, 0 if cat == "part" else re.I)) for label, p in entries]
    for cat, entries in TAG_VOCAB.items()
}


def vocab_tags(paper: dict, full_text: str, min_mentions: int = 2) -> dict[str, list[str]]:
    """Tag if mentioned in title/abstract, or at least `min_mentions` times anywhere."""
    head = f"{paper['title']} {paper['abstract']}"
    tags = {}
    for cat, entries in _COMPILED_VOCAB.items():
        found = []
        for label, rx in entries:
            if rx.search(head) or len(rx.findall(full_text)) >= min_mentions:
                found.append(label)
        tags[cat] = found
    return tags


def auto_keywords(doc_texts: list[str], top_n: int = 8) -> list[list[str]]:
    """Corpus-relative TF-IDF keyphrases per paper (unigrams + bigrams)."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    n = len(doc_texts)
    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words=list(STOPWORDS | {"cell", "cells", "gene", "genes", "model", "results", "system", "systems"}),
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9-]{2,}\b",
        sublinear_tf=True,
        min_df=2 if n >= 10 else 1,
        max_df=0.5 if n >= 10 else 1.0,
    )
    X = vec.fit_transform(doc_texts)
    terms = np.array(vec.get_feature_names_out())
    out = []
    for i in range(n):
        row = X.getrow(i)
        if row.nnz == 0:
            out.append([])
            continue
        order = np.argsort(-row.data)
        picked: list[str] = []
        for t in terms[row.indices[order]]:
            if any(t in p or p in t for p in picked):  # drop "toggle" if "toggle switch" is in, etc.
                continue
            picked.append(t)
            if len(picked) == top_n:
                break
        out.append(picked)
    return out

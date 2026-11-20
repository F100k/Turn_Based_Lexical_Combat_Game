"""Text normalisation and input validation (copied from notebook Section 2.1)."""


def clean_text(text: str) -> str:
    """Lowercase, replace non-letters with spaces, collapse spaces ('data-base' -> 'data base')."""
    text = str(text).lower().strip()
    text = "".join(ch if (ch.isalpha() or ch.isspace()) else " " for ch in text)
    return " ".join(text.split())


def duplicate_key(text: str) -> str:
    """Key that ignores word order and simple plurals ('business logics' == 'logic business')."""
    words = []
    for word in clean_text(text).split():
        if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        words.append(word)
    return " ".join(sorted(words))


def word_keys(text: str) -> set:
    """Normalised word keys ('ceos' -> 'ceo') used for target-word checks."""
    return {duplicate_key(w) for w in clean_text(text).split()}


def is_generic_term(cleaned_guess: str, generic_terms) -> bool:
    """True for broad field labels such as 'science' or 'business'."""
    return (cleaned_guess in generic_terms
            or duplicate_key(cleaned_guess) in {duplicate_key(t) for t in generic_terms})


def domain_term_pool(domain: str, examples: dict, vocab: dict, generic_terms) -> list:
    """De-duplicated, non-generic in-domain terms for one domain."""
    out, seen = [], set()
    for term in list(examples.get(domain, [])) + list(vocab.get(domain, [])):
        cleaned = clean_text(term)
        if cleaned and cleaned not in seen and not is_generic_term(cleaned, generic_terms):
            seen.add(cleaned)
            out.append(cleaned)
    return out


def validate_player_input(guess: str, target: str, used_guesses, block_target_word: bool = True) -> tuple:
    """Clean and validate a 1-3 word guess. Returns (is_valid, cleaned, reason)."""
    cleaned = clean_text(guess)
    target_clean = clean_text(target)
    if not cleaned:
        return False, cleaned, "Please enter 1–3 English words."
    if not 1 <= len(cleaned.split()) <= 3:
        return False, cleaned, "Your guess must contain between 1 and 3 words."
    if duplicate_key(cleaned) == duplicate_key(target_clean):
        return False, cleaned, "Your guess cannot be the target word itself."
    if block_target_word and (word_keys(cleaned) & word_keys(target_clean)):
        return False, cleaned, "Your guess cannot contain the target word."
    if cleaned in used_guesses:
        return False, cleaned, f"'{cleaned}' was already used in this battle."
    if duplicate_key(cleaned) in {duplicate_key(u) for u in used_guesses}:
        return False, cleaned, f"'{cleaned}' is too similar to a guess already used in this battle."
    return True, cleaned, ""

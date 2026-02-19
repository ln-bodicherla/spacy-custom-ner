"""
Entity Rules - Rule-based entity recognition using spaCy's Matcher and PhraseMatcher.

Provides pattern-based entity extraction through regex patterns, exact/fuzzy
phrase matching, and dependency-based syntactic patterns. Includes logic for
merging rule-based predictions with statistical model predictions with
configurable priority resolution.
"""

import re
import logging
from typing import Any

import spacy
from spacy.tokens import Doc, Span
from spacy.matcher import Matcher, PhraseMatcher
from spacy.language import Language
import yaml

logger = logging.getLogger(__name__)


class EntityRules:
    """
    Rule-based entity recognition combining multiple matching strategies.

    Supports token-level patterns (Matcher), phrase-level patterns (PhraseMatcher),
    regex patterns, and syntactic dependency patterns. Provides a merge mechanism
    for combining rule-based and statistical NER predictions.
    """

    def __init__(
        self,
        nlp: Language,
        config_path: str = "configs/config.yaml",
    ):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.nlp = nlp
        self.matcher = Matcher(nlp.vocab)
        self.phrase_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
        self.regex_patterns: list[dict[str, Any]] = []
        self.dependency_patterns: list[dict[str, Any]] = []

        self._pattern_labels: dict[str, str] = {}

    def add_regex_patterns(
        self, patterns: list[dict[str, str]]
    ) -> int:
        """
        Add regex-based entity patterns.

        Each pattern dict should contain 'pattern' (regex string) and
        'label' (entity type). Patterns are applied during extraction
        and matched spans are assigned the given label.
        """
        added = 0
        for pat in patterns:
            regex_str = pat.get("pattern", "")
            label = pat.get("label", "")

            if not regex_str or not label:
                continue

            try:
                re.compile(regex_str)
                self.regex_patterns.append({
                    "pattern": regex_str,
                    "label": label,
                    "flags": pat.get("flags", 0),
                })
                added += 1
            except re.error as e:
                logger.warning("Invalid regex pattern '%s': %s", regex_str, str(e))

        logger.info("Added %d regex patterns", added)
        return added

    def add_phrase_patterns(
        self,
        phrases: dict[str, list[str]],
        attr: str = "LOWER",
    ) -> int:
        """
        Add exact and fuzzy phrase matching patterns.

        Accepts a dictionary mapping entity labels to lists of phrases.
        Uses spaCy's PhraseMatcher for efficient multi-pattern matching.
        """
        total_added = 0

        for label, phrase_list in phrases.items():
            if not phrase_list:
                continue

            pattern_name = f"phrase_{label}"
            docs = list(self.nlp.pipe(phrase_list))

            self.phrase_matcher.add(pattern_name, docs)
            self._pattern_labels[pattern_name] = label
            total_added += len(phrase_list)

        logger.info("Added %d phrase patterns across %d labels",
                     total_added, len(phrases))
        return total_added

    def add_token_patterns(
        self, patterns: list[dict[str, Any]]
    ) -> int:
        """
        Add token-level patterns using spaCy's Matcher syntax.

        Each pattern dict should contain:
        - 'label': entity type to assign
        - 'pattern': list of token attribute dicts (spaCy Matcher format)
        """
        added = 0
        for i, pat in enumerate(patterns):
            label = pat.get("label", "")
            token_pattern = pat.get("pattern", [])

            if not label or not token_pattern:
                continue

            pattern_name = f"token_{label}_{i}"
            self.matcher.add(pattern_name, [token_pattern])
            self._pattern_labels[pattern_name] = label
            added += 1

        logger.info("Added %d token patterns", added)
        return added

    def add_dependency_patterns(
        self, patterns: list[dict[str, Any]]
    ) -> int:
        """
        Add syntactic dependency-based entity patterns.

        Patterns specify head token attributes and dependency relations
        to match, enabling extraction of entities based on syntactic
        structure rather than surface patterns.
        """
        added = 0
        for pat in patterns:
            label = pat.get("label", "")
            head_attrs = pat.get("head", {})
            dep_relation = pat.get("dep", "")
            child_attrs = pat.get("child", {})

            if not label or not dep_relation:
                continue

            self.dependency_patterns.append({
                "label": label,
                "head": head_attrs,
                "dep": dep_relation,
                "child": child_attrs,
                "include_subtree": pat.get("include_subtree", False),
            })
            added += 1

        logger.info("Added %d dependency patterns", added)
        return added

    def extract_by_regex(self, doc: Doc) -> list[dict[str, Any]]:
        """Apply regex patterns to find entities in the document text."""
        matches = []
        text = doc.text

        for pat_info in self.regex_patterns:
            pattern = pat_info["pattern"]
            label = pat_info["label"]
            flags = pat_info.get("flags", 0)

            for match in re.finditer(pattern, text, flags=flags):
                start_char = match.start()
                end_char = match.end()

                span = doc.char_span(
                    start_char, end_char,
                    label=label,
                    alignment_mode="expand",
                )

                if span is not None and len(span.text.strip()) > 0:
                    matches.append({
                        "text": span.text,
                        "label": label,
                        "start": start_char,
                        "end": end_char,
                        "start_token": span.start,
                        "end_token": span.end,
                        "source": "regex",
                    })

        return matches

    def extract_by_phrase(self, doc: Doc) -> list[dict[str, Any]]:
        """Apply phrase matcher patterns to find entities."""
        matches = []
        phrase_matches = self.phrase_matcher(doc)

        for match_id, start, end in phrase_matches:
            pattern_name = self.nlp.vocab.strings[match_id]
            label = self._pattern_labels.get(pattern_name, "UNKNOWN")
            span = doc[start:end]

            matches.append({
                "text": span.text,
                "label": label,
                "start": span.start_char,
                "end": span.end_char,
                "start_token": start,
                "end_token": end,
                "source": "phrase",
            })

        return matches

    def extract_by_token_pattern(self, doc: Doc) -> list[dict[str, Any]]:
        """Apply token-level matcher patterns to find entities."""
        matches = []
        token_matches = self.matcher(doc)

        for match_id, start, end in token_matches:
            pattern_name = self.nlp.vocab.strings[match_id]
            label = self._pattern_labels.get(pattern_name, "UNKNOWN")
            span = doc[start:end]

            matches.append({
                "text": span.text,
                "label": label,
                "start": span.start_char,
                "end": span.end_char,
                "start_token": start,
                "end_token": end,
                "source": "token_pattern",
            })

        return matches

    def extract_by_dependency(self, doc: Doc) -> list[dict[str, Any]]:
        """
        Apply dependency-based patterns to find entities.

        Traverses the dependency tree looking for specific head-child
        relationships matching the configured patterns.
        """
        matches = []

        for pat in self.dependency_patterns:
            label = pat["label"]
            dep_relation = pat["dep"]
            head_attrs = pat["head"]
            child_attrs = pat["child"]
            include_subtree = pat.get("include_subtree", False)

            for token in doc:
                if token.dep_ != dep_relation:
                    continue

                head = token.head
                if not self._token_matches_attrs(head, head_attrs):
                    continue
                if not self._token_matches_attrs(token, child_attrs):
                    continue

                if include_subtree:
                    subtree = sorted(token.subtree, key=lambda t: t.i)
                    if subtree:
                        start = subtree[0].i
                        end = subtree[-1].i + 1
                        span = doc[start:end]
                    else:
                        span = doc[token.i:token.i + 1]
                else:
                    span = doc[token.i:token.i + 1]

                matches.append({
                    "text": span.text,
                    "label": label,
                    "start": span.start_char,
                    "end": span.end_char,
                    "start_token": span.start,
                    "end_token": span.end,
                    "source": "dependency",
                })

        return matches

    def _token_matches_attrs(self, token, attrs: dict) -> bool:
        """Check if a token matches the specified attribute constraints."""
        if not attrs:
            return True

        for attr, value in attrs.items():
            if attr == "POS" and token.pos_ != value:
                return False
            elif attr == "DEP" and token.dep_ != value:
                return False
            elif attr == "LEMMA" and token.lemma_.lower() != value.lower():
                return False
            elif attr == "TEXT" and token.text.lower() != value.lower():
                return False
            elif attr == "TAG" and token.tag_ != value:
                return False

        return True

    def extract_all(self, doc: Doc) -> list[dict[str, Any]]:
        """Run all rule-based extraction methods and combine results."""
        all_matches = []
        all_matches.extend(self.extract_by_regex(doc))
        all_matches.extend(self.extract_by_phrase(doc))
        all_matches.extend(self.extract_by_token_pattern(doc))
        all_matches.extend(self.extract_by_dependency(doc))

        deduplicated = self._deduplicate_matches(all_matches)
        return deduplicated

    def _deduplicate_matches(
        self, matches: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Remove duplicate and overlapping matches, keeping the longest."""
        if not matches:
            return []

        matches.sort(key=lambda m: (m["start"], -(m["end"] - m["start"])))
        result = []
        last_end = -1

        for match in matches:
            if match["start"] >= last_end:
                result.append(match)
                last_end = match["end"]
            elif match["end"] - match["start"] > result[-1]["end"] - result[-1]["start"]:
                result[-1] = match
                last_end = match["end"]

        return result

    def combine_with_statistical(
        self,
        doc: Doc,
        rule_entities: list[dict[str, Any]],
        priority: str = "statistical",
    ) -> Doc:
        """
        Merge rule-based and statistical NER predictions.

        Resolves conflicts between the two sources using the specified
        priority strategy:
        - 'statistical': prefer model predictions when there's overlap
        - 'rules': prefer rule-based predictions when there's overlap
        - 'longest': keep whichever span is longer
        """
        stat_entities = []
        for ent in doc.ents:
            stat_entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
                "start_token": ent.start,
                "end_token": ent.end,
                "source": "statistical",
            })

        all_entities = stat_entities + rule_entities

        resolved = self._resolve_conflicts(all_entities, priority)

        new_ents = []
        for ent_info in resolved:
            span = doc.char_span(
                ent_info["start"], ent_info["end"],
                label=ent_info["label"],
                alignment_mode="expand",
            )
            if span is not None:
                new_ents.append(span)

        new_ents.sort(key=lambda s: s.start)
        filtered_ents = []
        last_end = -1
        for ent in new_ents:
            if ent.start >= last_end:
                filtered_ents.append(ent)
                last_end = ent.end

        try:
            doc.ents = filtered_ents
        except ValueError:
            logger.warning("Failed to set merged entities, keeping original")

        return doc

    def _resolve_conflicts(
        self,
        entities: list[dict[str, Any]],
        priority: str,
    ) -> list[dict[str, Any]]:
        """Resolve overlapping entity predictions based on priority strategy."""
        if not entities:
            return []

        entities.sort(key=lambda e: (e["start"], -(e["end"] - e["start"])))
        resolved = []

        for entity in entities:
            overlapping = None
            for i, existing in enumerate(resolved):
                if entity["start"] < existing["end"] and existing["start"] < entity["end"]:
                    overlapping = i
                    break

            if overlapping is None:
                resolved.append(entity)
                continue

            existing = resolved[overlapping]

            if priority == "statistical":
                if entity["source"] == "statistical":
                    resolved[overlapping] = entity
            elif priority == "rules":
                if entity["source"] != "statistical":
                    resolved[overlapping] = entity
            elif priority == "longest":
                entity_len = entity["end"] - entity["start"]
                existing_len = existing["end"] - existing["start"]
                if entity_len > existing_len:
                    resolved[overlapping] = entity

        return resolved

    def get_pattern_summary(self) -> dict[str, Any]:
        """Return a summary of all registered patterns."""
        return {
            "regex_patterns": len(self.regex_patterns),
            "phrase_patterns": len(self.phrase_matcher),
            "token_patterns": len(self.matcher),
            "dependency_patterns": len(self.dependency_patterns),
            "total": (
                len(self.regex_patterns)
                + len(self.phrase_matcher)
                + len(self.matcher)
                + len(self.dependency_patterns)
            ),
            "labels": list(set(
                [p["label"] for p in self.regex_patterns]
                + list(self._pattern_labels.values())
                + [p["label"] for p in self.dependency_patterns]
            )),
        }

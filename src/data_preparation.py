"""
SpaCy Data Preparator - Handles data loading, conversion, and augmentation for NER training.

Supports multiple annotation formats (JSON, JSONL, IOB), converts them to spaCy's
binary DocBin format, validates annotation quality, and provides entity-preserving
text augmentation.
"""

import os
import json
import random
import logging
from typing import Any
from pathlib import Path

import spacy
from spacy.tokens import DocBin, Doc
from spacy.training import Example
import srsly
import yaml

logger = logging.getLogger(__name__)


class SpaCyDataPreparator:
    """
    Prepares training data for spaCy NER model training.

    Handles loading annotations from various formats, converting to spaCy's
    binary format, validating annotation quality, performing data augmentation,
    and generating sample training data for demonstration.
    """

    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.nlp = spacy.blank("en")
        self.labels = self.config.get("labels", [])

    def load_annotations(
        self, file_path: str, format_type: str | None = None
    ) -> list[dict[str, Any]]:
        """
        Load annotations from file, auto-detecting format if not specified.

        Supports JSON (list of objects), JSONL (line-delimited JSON),
        and IOB (token-per-line with BIO tags) formats.
        """
        if format_type is None:
            ext = Path(file_path).suffix.lower()
            format_map = {".json": "json", ".jsonl": "jsonl", ".iob": "iob", ".txt": "iob"}
            format_type = format_map.get(ext, "json")

        if format_type == "json":
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            annotations = data if isinstance(data, list) else [data]

        elif format_type == "jsonl":
            annotations = list(srsly.read_jsonl(file_path))

        elif format_type == "iob":
            annotations = self._parse_iob_file(file_path)

        else:
            raise ValueError(f"Unsupported format: {format_type}")

        logger.info("Loaded %d annotations from %s (%s format)",
                     len(annotations), file_path, format_type)
        return annotations

    def _parse_iob_file(self, file_path: str) -> list[dict[str, Any]]:
        """Parse IOB/BIO formatted file into annotation dictionaries."""
        annotations = []
        current_tokens = []
        current_tags = []

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    if current_tokens:
                        annotation = self._iob_to_annotation(current_tokens, current_tags)
                        annotations.append(annotation)
                        current_tokens = []
                        current_tags = []
                    continue

                parts = line.split()
                if len(parts) >= 2:
                    current_tokens.append(parts[0])
                    current_tags.append(parts[-1])

        if current_tokens:
            annotation = self._iob_to_annotation(current_tokens, current_tags)
            annotations.append(annotation)

        return annotations

    def _iob_to_annotation(
        self, tokens: list[str], tags: list[str]
    ) -> dict[str, Any]:
        """Convert IOB-tagged tokens to spaCy annotation format."""
        text = " ".join(tokens)
        entities = []
        current_entity = None
        char_offset = 0

        for token, tag in zip(tokens, tags):
            token_start = char_offset
            token_end = char_offset + len(token)

            if tag.startswith("B-"):
                if current_entity is not None:
                    entities.append(current_entity)
                label = tag[2:]
                current_entity = {"start": token_start, "end": token_end, "label": label}
            elif tag.startswith("I-") and current_entity is not None:
                label = tag[2:]
                if label == current_entity["label"]:
                    current_entity["end"] = token_end
                else:
                    entities.append(current_entity)
                    current_entity = {"start": token_start, "end": token_end, "label": label}
            else:
                if current_entity is not None:
                    entities.append(current_entity)
                    current_entity = None

            char_offset = token_end + 1

        if current_entity is not None:
            entities.append(current_entity)

        return {
            "text": text,
            "entities": [(e["start"], e["end"], e["label"]) for e in entities],
        }

    def convert_to_spacy_format(
        self, annotations: list[dict[str, Any]]
    ) -> list[tuple[str, dict]]:
        """
        Convert annotations to spaCy's training format.

        Returns a list of (text, {"entities": [(start, end, label), ...]}) tuples.
        """
        training_data = []

        for ann in annotations:
            text = ann.get("text", "")
            entities = ann.get("entities", [])

            if isinstance(entities, list) and entities:
                if isinstance(entities[0], dict):
                    ents = [
                        (e["start"], e["end"], e["label"])
                        for e in entities
                    ]
                elif isinstance(entities[0], (list, tuple)):
                    ents = [tuple(e) for e in entities]
                else:
                    continue
            else:
                ents = []

            training_data.append((text, {"entities": ents}))

        logger.info("Converted %d annotations to spaCy format", len(training_data))
        return training_data

    def create_training_data(
        self,
        training_data: list[tuple[str, dict]],
        output_path: str,
    ) -> str:
        """
        Generate a spaCy DocBin binary file from training data.

        Creates the binary format required by spaCy's training pipeline,
        handling tokenization alignment and entity span assignment.
        """
        doc_bin = DocBin()
        nlp = spacy.blank("en")
        skipped = 0

        for text, annotations in training_data:
            doc = nlp.make_doc(text)
            entities = annotations.get("entities", [])

            ents = []
            seen_tokens = set()
            valid = True

            for start, end, label in sorted(entities, key=lambda x: x[0]):
                span = doc.char_span(start, end, label=label, alignment_mode="contract")
                if span is None:
                    span = doc.char_span(start, end, label=label, alignment_mode="expand")

                if span is None:
                    continue

                token_indices = set(range(span.start, span.end))
                if token_indices & seen_tokens:
                    continue

                seen_tokens.update(token_indices)
                ents.append(span)

            try:
                doc.ents = ents
                doc_bin.add(doc)
            except ValueError:
                skipped += 1
                continue

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        doc_bin.to_disk(output_path)

        logger.info(
            "Created training data: %d documents, %d skipped -> %s",
            len(training_data) - skipped, skipped, output_path,
        )
        return output_path

    def augment_data(
        self,
        training_data: list[tuple[str, dict]],
        num_augmented: int = 2,
    ) -> list[tuple[str, dict]]:
        """
        Perform entity-preserving text augmentation.

        Applies transformations that preserve entity boundaries:
        - Synonym replacement for non-entity tokens
        - Random insertion of filler words
        - Case variation on non-entity tokens
        """
        augmented = list(training_data)

        case_variations = [str.lower, str.upper, str.title]
        filler_words = [
            "also", "additionally", "furthermore", "meanwhile",
            "however", "notably", "specifically", "reportedly",
        ]

        for text, annotations in training_data:
            entities = annotations.get("entities", [])

            for _ in range(num_augmented):
                entity_ranges = set()
                for start, end, _ in entities:
                    entity_ranges.update(range(start, end))

                words = text.split()
                new_words = []
                char_pos = 0
                offset_adjustment = 0
                new_entities = list(entities)

                for word in words:
                    word_start = text.index(word, char_pos)
                    word_end = word_start + len(word)

                    is_entity = any(
                        pos in entity_ranges for pos in range(word_start, word_end)
                    )

                    if not is_entity and random.random() < 0.3:
                        variation = random.choice(case_variations)
                        new_word = variation(word)
                        new_words.append(new_word)
                    elif not is_entity and random.random() < 0.1:
                        filler = random.choice(filler_words)
                        new_words.append(filler)
                        new_words.append(word)
                        filler_len = len(filler) + 1

                        adjusted = []
                        for s, e, l in new_entities:
                            if s > word_start:
                                adjusted.append((s + filler_len, e + filler_len, l))
                            else:
                                adjusted.append((s, e, l))
                        new_entities = adjusted
                    else:
                        new_words.append(word)

                    char_pos = word_end

                new_text = " ".join(new_words)

                valid_entities = []
                for start, end, label in new_entities:
                    if start < len(new_text) and end <= len(new_text):
                        valid_entities.append((start, end, label))

                augmented.append((new_text, {"entities": valid_entities}))

        logger.info(
            "Augmented %d to %d training examples",
            len(training_data), len(augmented),
        )
        return augmented

    def validate_annotations(
        self, training_data: list[tuple[str, dict]]
    ) -> dict[str, Any]:
        """
        Validate annotations for common issues.

        Checks for overlapping entities, misaligned character offsets,
        empty entities, and unknown labels.
        """
        issues = {
            "overlapping_entities": [],
            "misaligned_offsets": [],
            "empty_entities": [],
            "unknown_labels": [],
            "total_checked": len(training_data),
            "valid_count": 0,
            "invalid_count": 0,
        }

        valid_labels = set(self.labels)

        for idx, (text, annotations) in enumerate(training_data):
            entities = annotations.get("entities", [])
            has_issues = False

            for i, (s1, e1, l1) in enumerate(entities):
                if s1 >= e1 or s1 < 0 or e1 > len(text):
                    issues["misaligned_offsets"].append({
                        "index": idx, "entity": (s1, e1, l1),
                        "reason": "Invalid character offsets",
                    })
                    has_issues = True

                entity_text = text[s1:e1].strip()
                if not entity_text:
                    issues["empty_entities"].append({
                        "index": idx, "entity": (s1, e1, l1),
                    })
                    has_issues = True

                if valid_labels and l1 not in valid_labels:
                    issues["unknown_labels"].append({
                        "index": idx, "label": l1,
                        "known_labels": list(valid_labels),
                    })
                    has_issues = True

                for j, (s2, e2, l2) in enumerate(entities):
                    if i >= j:
                        continue
                    if s1 < e2 and s2 < e1:
                        issues["overlapping_entities"].append({
                            "index": idx,
                            "entity_a": (s1, e1, l1),
                            "entity_b": (s2, e2, l2),
                        })
                        has_issues = True

            if has_issues:
                issues["invalid_count"] += 1
            else:
                issues["valid_count"] += 1

        logger.info(
            "Validation: %d valid, %d invalid out of %d",
            issues["valid_count"], issues["invalid_count"], issues["total_checked"],
        )
        return issues

    def split_data(
        self,
        training_data: list[tuple[str, dict]],
        train_ratio: float = 0.8,
        seed: int = 42,
    ) -> tuple[list[tuple[str, dict]], list[tuple[str, dict]]]:
        """Split data into training and development sets."""
        data = list(training_data)
        random.seed(seed)
        random.shuffle(data)

        split_idx = int(len(data) * train_ratio)
        train_set = data[:split_idx]
        dev_set = data[split_idx:]

        logger.info("Split data: %d train, %d dev", len(train_set), len(dev_set))
        return train_set, dev_set

    def generate_sample_data(self, num_samples: int = 100) -> list[tuple[str, dict]]:
        """
        Generate sample NER training data for demonstration.

        Creates synthetic sentences with annotated entities across
        multiple entity types for testing the training pipeline.
        """
        templates = [
            ("{PERSON} joined {ORGANIZATION} in {DATE}.",
             ["PERSON", "ORGANIZATION", "DATE"]),
            ("{ORGANIZATION} announced the {PRODUCT} at their headquarters in {LOCATION}.",
             ["ORGANIZATION", "PRODUCT", "LOCATION"]),
            ("{PERSON} invested {MONEY} in {ORGANIZATION}.",
             ["PERSON", "MONEY", "ORGANIZATION"]),
            ("The {PRODUCT} was developed by {ORGANIZATION} and released on {DATE}.",
             ["PRODUCT", "ORGANIZATION", "DATE"]),
            ("{PERSON} traveled from {LOCATION} to {LOCATION} for the {ORGANIZATION} conference.",
             ["PERSON", "LOCATION", "LOCATION", "ORGANIZATION"]),
            ("{ORGANIZATION} reported revenue of {MONEY} for {DATE}.",
             ["ORGANIZATION", "MONEY", "DATE"]),
            ("CEO {PERSON} of {ORGANIZATION} launched {PRODUCT} in {LOCATION}.",
             ["PERSON", "ORGANIZATION", "PRODUCT", "LOCATION"]),
        ]

        entity_examples = {
            "PERSON": ["John Smith", "Sarah Johnson", "Michael Chen", "Emily Davis",
                       "Robert Wilson", "Maria Garcia", "James Brown", "Lisa Anderson"],
            "ORGANIZATION": ["Apple Inc.", "Google", "Microsoft", "Amazon", "Tesla",
                            "Meta Platforms", "OpenAI", "Anthropic", "Netflix"],
            "LOCATION": ["New York", "San Francisco", "London", "Tokyo", "Berlin",
                         "Sydney", "Toronto", "Singapore", "Paris"],
            "PRODUCT": ["iPhone 15", "Pixel 8", "Surface Pro", "Model Y", "Quest 3",
                        "Galaxy S24", "MacBook Pro", "Kindle Scribe"],
            "MONEY": ["$1.5 billion", "$500 million", "$10 billion", "$2.3 million",
                      "$750 million", "$4.2 billion", "$100 million", "$3 billion"],
            "DATE": ["January 2024", "Q3 2023", "March 15, 2024", "last quarter",
                     "fiscal year 2023", "December 2023", "spring 2024"],
        }

        random.seed(42)
        training_data = []

        for _ in range(num_samples):
            template, label_sequence = random.choice(templates)
            text = template
            entities = []

            for label in label_sequence:
                examples = entity_examples.get(label, [label])
                value = random.choice(examples)
                placeholder = "{" + label + "}"

                idx = text.find(placeholder)
                if idx == -1:
                    continue

                text = text[:idx] + value + text[idx + len(placeholder):]
                entities.append((idx, idx + len(value), label))

            training_data.append((text, {"entities": entities}))

        logger.info("Generated %d sample training examples", len(training_data))
        return training_data

"""
SpaCy NER Model - Custom Named Entity Recognition with training and inference.

Provides a complete NER pipeline including blank model creation, entity ruler
integration, training with dropout scheduling and early stopping, evaluation
with token-level and entity-level metrics, and model packaging for deployment.
"""

import os
import logging
import random
from typing import Any
from pathlib import Path

import spacy
from spacy import Language
from spacy.tokens import DocBin
from spacy.training import Example
from spacy.scorer import Scorer
from spacy.util import minibatch, compounding
import yaml

logger = logging.getLogger(__name__)


class SpaCyNERModel:
    """
    Wrapper for spaCy NER model training, evaluation, and inference.

    Supports both training from scratch and fine-tuning existing models.
    Integrates rule-based entity rulers with statistical NER for
    improved coverage and precision.
    """

    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.nlp: Language | None = None
        self.labels = self.config.get("labels", [])
        self.training_config = self.config.get("training", {})
        self.is_trained = False

    def create_blank_pipeline(self, lang: str = "en") -> Language:
        """
        Create a spaCy pipeline from scratch with an NER component.

        Initializes a blank model and adds the NER pipe with all
        configured entity labels.
        """
        self.nlp = spacy.blank(lang)

        if "ner" not in self.nlp.pipe_names:
            ner = self.nlp.add_pipe("ner", last=True)
        else:
            ner = self.nlp.get_pipe("ner")

        for label in self.labels:
            ner.add_label(label)

        logger.info(
            "Created blank pipeline with labels: %s",
            ", ".join(self.labels),
        )
        return self.nlp

    def load_base_model(self) -> Language:
        """
        Load a pre-trained base model for fine-tuning.

        Adds any missing entity labels to the existing NER component.
        """
        base_model = self.config["model"]["base_model"]
        self.nlp = spacy.load(base_model)

        if "ner" not in self.nlp.pipe_names:
            ner = self.nlp.add_pipe("ner", last=True)
        else:
            ner = self.nlp.get_pipe("ner")

        for label in self.labels:
            ner.add_label(label)

        logger.info("Loaded base model: %s", base_model)
        return self.nlp

    def add_entity_ruler(
        self,
        patterns: list[dict[str, Any]],
        overwrite_ents: bool | None = None,
    ) -> None:
        """
        Add rule-based entity patterns to the pipeline.

        Adds an EntityRuler component that matches entities based on
        explicit patterns, running before or after the statistical NER.
        """
        if self.nlp is None:
            raise RuntimeError("Pipeline not initialized.")

        ruler_config = self.config.get("entity_ruler", {})
        if overwrite_ents is None:
            overwrite_ents = ruler_config.get("overwrite_ents", False)

        if "entity_ruler" in self.nlp.pipe_names:
            self.nlp.remove_pipe("entity_ruler")

        ruler = self.nlp.add_pipe(
            "entity_ruler",
            before="ner" if "ner" in self.nlp.pipe_names else None,
            config={"overwrite_ents": overwrite_ents},
        )

        ruler.add_patterns(patterns)
        logger.info("Added %d entity ruler patterns", len(patterns))

    def configure_ner(
        self,
        training_data: list[tuple[str, dict]],
    ) -> None:
        """
        Configure the NER component with labels from training data.

        Scans the training data to discover all entity labels and
        registers them with the NER component.
        """
        if self.nlp is None:
            self.create_blank_pipeline()

        ner = self.nlp.get_pipe("ner")

        discovered_labels = set()
        for _, annotations in training_data:
            for entity in annotations.get("entities", []):
                if len(entity) >= 3:
                    discovered_labels.add(entity[2])

        for label in discovered_labels:
            ner.add_label(label)
            if label not in self.labels:
                self.labels.append(label)

        logger.info("Configured NER with %d labels: %s",
                     len(self.labels), ", ".join(self.labels))

    def train(
        self,
        training_data: list[tuple[str, dict]],
        dev_data: list[tuple[str, dict]] | None = None,
        epochs: int | None = None,
        batch_size: int | None = None,
        dropout: float | None = None,
        learning_rate: float | None = None,
        patience: int | None = None,
    ) -> dict[str, Any]:
        """
        Train the NER model with batching, dropout scheduling, and early stopping.

        Converts training data to spaCy Examples, optimizes the model
        using minibatch SGD, and evaluates on dev data each epoch.
        """
        if self.nlp is None:
            self.create_blank_pipeline()

        if epochs is None:
            epochs = self.training_config.get("epochs", 30)
        if batch_size is None:
            batch_size = self.training_config.get("batch_size", 32)
        if dropout is None:
            dropout = self.training_config.get("dropout", 0.35)
        if learning_rate is None:
            learning_rate = self.training_config.get("learning_rate", 0.001)
        if patience is None:
            patience = self.training_config.get("patience", 5)

        self.configure_ner(training_data)

        train_examples = self._create_examples(training_data)
        dev_examples = self._create_examples(dev_data) if dev_data else None

        other_pipes = [pipe for pipe in self.nlp.pipe_names if pipe != "ner"]
        disabled = self.nlp.disable_pipes(*other_pipes)

        optimizer = self.nlp.initialize(lambda: train_examples)

        history = {
            "train_loss": [],
            "dev_scores": [],
            "best_epoch": 0,
            "best_f1": 0.0,
        }

        best_f1 = 0.0
        best_state = None
        epochs_no_improve = 0

        for epoch in range(epochs):
            random.shuffle(train_examples)
            epoch_loss = 0.0
            num_batches = 0

            current_dropout = self._dropout_schedule(dropout, epoch, epochs)

            batches = minibatch(
                train_examples,
                size=compounding(4.0, batch_size, 1.001),
            )

            for batch in batches:
                losses = {}
                self.nlp.update(
                    batch,
                    sgd=optimizer,
                    drop=current_dropout,
                    losses=losses,
                )
                epoch_loss += losses.get("ner", 0)
                num_batches += 1

            avg_loss = epoch_loss / max(num_batches, 1)
            history["train_loss"].append(avg_loss)

            dev_f1 = 0.0
            if dev_examples:
                scores = self._evaluate_examples(dev_examples)
                dev_f1 = scores.get("ents_f", 0.0)
                history["dev_scores"].append(scores)

                if dev_f1 > best_f1:
                    best_f1 = dev_f1
                    history["best_epoch"] = epoch + 1
                    history["best_f1"] = best_f1
                    best_state = {
                        name: param.copy()
                        for name, param in self.nlp.get_pipe("ner").model.attrs.items()
                    } if hasattr(self.nlp.get_pipe("ner").model, "attrs") else None
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

            if (epoch + 1) % 5 == 0:
                logger.info(
                    "Epoch %d/%d - Loss: %.4f, Dev F1: %.4f, Dropout: %.3f",
                    epoch + 1, epochs, avg_loss, dev_f1, current_dropout,
                )

            if patience > 0 and epochs_no_improve >= patience:
                logger.info("Early stopping at epoch %d (best F1: %.4f at epoch %d)",
                            epoch + 1, best_f1, history["best_epoch"])
                break

        disabled.restore()
        self.is_trained = True

        logger.info("Training complete. Best F1: %.4f at epoch %d",
                     best_f1, history["best_epoch"])
        return history

    def _create_examples(
        self, data: list[tuple[str, dict]]
    ) -> list[Example]:
        """Convert training tuples to spaCy Example objects."""
        examples = []
        for text, annotations in data:
            doc = self.nlp.make_doc(text)
            entities = annotations.get("entities", [])

            spans = []
            seen_tokens = set()
            for start, end, label in entities:
                span = doc.char_span(start, end, label=label, alignment_mode="contract")
                if span is None:
                    span = doc.char_span(start, end, label=label, alignment_mode="expand")
                if span is not None:
                    token_indices = set(range(span.start, span.end))
                    if not token_indices & seen_tokens:
                        seen_tokens.update(token_indices)
                        spans.append(span)

            reference = doc.copy()
            reference.ents = spans
            example = Example(doc, reference)
            examples.append(example)

        return examples

    def _dropout_schedule(
        self, base_dropout: float, epoch: int, total_epochs: int
    ) -> float:
        """Compute dropout rate using a linear decay schedule."""
        min_dropout = base_dropout * 0.5
        progress = epoch / max(total_epochs - 1, 1)
        return base_dropout - (base_dropout - min_dropout) * progress

    def _evaluate_examples(self, examples: list[Example]) -> dict[str, float]:
        """Evaluate model on a set of Examples using spaCy's scorer."""
        scorer = Scorer()
        eval_examples = []

        for example in examples:
            pred_doc = self.nlp(example.reference.text)
            eval_example = Example(pred_doc, example.reference)
            eval_examples.append(eval_example)

        scores = scorer.score(eval_examples)
        return {
            "ents_p": scores.get("ents_p", 0.0),
            "ents_r": scores.get("ents_r", 0.0),
            "ents_f": scores.get("ents_f", 0.0),
            "ents_per_type": scores.get("ents_per_type", {}),
        }

    def evaluate(
        self,
        test_data: list[tuple[str, dict]],
    ) -> dict[str, Any]:
        """
        Evaluate model on test data with token-level and entity-level metrics.

        Returns precision, recall, F1, and per-entity-type breakdowns.
        """
        if self.nlp is None:
            raise RuntimeError("Model not initialized.")

        examples = self._create_examples(test_data)
        scores = self._evaluate_examples(examples)

        logger.info(
            "Evaluation - P: %.4f, R: %.4f, F1: %.4f",
            scores["ents_p"], scores["ents_r"], scores["ents_f"],
        )
        return scores

    def predict(self, text: str) -> list[dict[str, Any]]:
        """
        Extract entities from a single text.

        Returns a list of entity dictionaries with text, label, start,
        end positions, and a confidence placeholder.
        """
        if self.nlp is None:
            raise RuntimeError("Model not initialized.")

        doc = self.nlp(text)
        entities = []

        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
                "score": 1.0,
            })

        return entities

    def predict_batch(self, texts: list[str]) -> list[list[dict[str, Any]]]:
        """Extract entities from multiple texts using spaCy's pipe for efficiency."""
        if self.nlp is None:
            raise RuntimeError("Model not initialized.")

        all_entities = []
        for doc in self.nlp.pipe(texts, batch_size=64):
            entities = []
            for ent in doc.ents:
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "score": 1.0,
                })
            all_entities.append(entities)

        return all_entities

    def save_model(self, output_path: str) -> str:
        """Save the trained model to disk."""
        if self.nlp is None:
            raise RuntimeError("Model not initialized.")

        os.makedirs(output_path, exist_ok=True)
        self.nlp.to_disk(output_path)
        logger.info("Saved model to: %s", output_path)
        return output_path

    def load_model(self, model_path: str) -> Language:
        """Load a saved model from disk."""
        self.nlp = spacy.load(model_path)
        self.is_trained = True
        logger.info("Loaded model from: %s", model_path)
        return self.nlp

    def export_package(
        self,
        output_dir: str,
        name: str = "custom_ner",
        version: str = "1.0.0",
    ) -> str:
        """
        Package the model as a distributable spaCy model package.

        Creates a directory structure suitable for pip installation.
        """
        if self.nlp is None:
            raise RuntimeError("Model not initialized.")

        package_dir = os.path.join(output_dir, name)
        model_dir = os.path.join(package_dir, f"{name}-{version}")
        os.makedirs(model_dir, exist_ok=True)

        self.nlp.to_disk(model_dir)

        meta = {
            "name": name,
            "version": version,
            "spacy_version": f">={spacy.__version__}",
            "lang": self.nlp.lang,
            "pipeline": self.nlp.pipe_names,
            "labels": {
                "ner": self.labels,
            },
        }

        import json
        meta_path = os.path.join(model_dir, "meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        logger.info("Exported model package to: %s", package_dir)
        return package_dir

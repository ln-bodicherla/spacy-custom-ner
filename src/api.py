"""
FastAPI application for Custom SpaCy NER.

Provides REST endpoints for named entity extraction, batch processing,
annotated HTML output, and model metadata.
"""

import os
import logging
from typing import Any
from contextlib import asynccontextmanager

import spacy
from spacy import displacy
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.model import SpaCyNERModel
from src.rules import EntityRules
from src.data_preparation import SpaCyDataPreparator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

CONFIG_PATH = os.getenv("CONFIG_PATH", "configs/config.yaml")

ner_model: SpaCyNERModel | None = None
entity_rules: EntityRules | None = None


def _setup_default_rules(rules: EntityRules) -> None:
    """Configure default entity matching rules."""
    rules.add_regex_patterns([
        {"pattern": r"\$[\d,]+(?:\.\d{1,2})?\s*(?:billion|million|thousand)?", "label": "MONEY"},
        {"pattern": r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", "label": "DATE"},
        {"pattern": r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b", "label": "DATE"},
        {"pattern": r"\b(?:Q[1-4]|H[12])\s+\d{4}\b", "label": "DATE"},
    ])

    rules.add_phrase_patterns({
        "ORGANIZATION": [
            "Apple Inc.", "Google LLC", "Microsoft Corporation",
            "Amazon.com", "Meta Platforms", "Tesla Inc.",
            "OpenAI", "Anthropic", "Netflix Inc.",
        ],
        "PRODUCT": [
            "iPhone", "iPad", "MacBook", "Pixel", "Surface Pro",
            "Galaxy", "Kindle", "Echo", "Quest",
        ],
    })

    rules.add_token_patterns([
        {
            "label": "MONEY",
            "pattern": [
                {"IS_CURRENCY": True},
                {"LIKE_NUM": True},
                {"LOWER": {"IN": ["billion", "million", "thousand"]}, "OP": "?"},
            ],
        },
        {
            "label": "PERSON",
            "pattern": [
                {"POS": "PROPN"},
                {"POS": "PROPN"},
            ],
        },
    ])

    logger.info("Loaded default entity rules")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle: load model and rules on startup."""
    global ner_model, entity_rules

    logger.info("Initializing SpaCy NER service")

    ner_model = SpaCyNERModel(config_path=CONFIG_PATH)

    model_path = os.getenv("MODEL_PATH", "models/custom_ner")
    if os.path.isdir(model_path):
        ner_model.load_model(model_path)
        logger.info("Loaded custom NER model from: %s", model_path)
    else:
        ner_model.load_base_model()
        logger.info("Loaded base model (no custom model found)")

    entity_rules = EntityRules(ner_model.nlp, config_path=CONFIG_PATH)
    _setup_default_rules(entity_rules)

    logger.info("SpaCy NER service initialized")
    yield
    logger.info("Shutting down SpaCy NER service")


app = FastAPI(
    title="Custom NER with SpaCy",
    description=(
        "Lightweight, production-ready Named Entity Recognition using SpaCy. "
        "Combines statistical NER with rule-based entity matching for "
        "high-coverage, low-latency entity extraction."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    """Request schema for entity prediction."""
    text: str = Field(..., min_length=1, max_length=100000, description="Input text")
    use_rules: bool = Field(default=True, description="Apply rule-based matching in addition to NER")
    priority: str = Field(default="statistical", description="Conflict resolution: statistical, rules, longest")


class EntityResult(BaseModel):
    """Single entity extraction result."""
    text: str
    label: str
    start: int
    end: int
    score: float


class PredictResponse(BaseModel):
    """Response schema for entity prediction."""
    text: str
    entities: list[EntityResult]
    num_entities: int


class BatchPredictRequest(BaseModel):
    """Request schema for batch entity prediction."""
    texts: list[str] = Field(..., min_length=1, max_length=1000, description="List of input texts")
    use_rules: bool = Field(default=True)


class BatchPredictResponse(BaseModel):
    """Response schema for batch entity prediction."""
    results: list[PredictResponse]
    total_entities: int


class AnnotateRequest(BaseModel):
    """Request schema for annotated HTML output."""
    text: str = Field(..., min_length=1, max_length=100000)
    colors: dict[str, str] | None = Field(default=None, description="Custom colors per label")


class LabelsResponse(BaseModel):
    """Response schema for supported entity labels."""
    labels: list[str]
    num_labels: int
    rule_patterns: dict[str, Any]


class TrainRequest(BaseModel):
    """Request schema for model training."""
    use_sample_data: bool = Field(default=True, description="Use sample data for demo training")
    epochs: int = Field(default=10, ge=1, le=100)
    num_samples: int = Field(default=100, ge=10, le=10000)


class TrainResponse(BaseModel):
    """Response schema for model training."""
    status: str
    epochs_trained: int
    best_f1: float
    labels: list[str]


class HealthResponse(BaseModel):
    """Response schema for health check."""
    status: str
    model_loaded: bool
    pipeline: list[str]
    num_labels: int


@app.post("/predict", response_model=PredictResponse)
async def predict_entities(request: PredictRequest):
    """
    Extract named entities from the provided text.

    Combines statistical NER predictions with optional rule-based matching.
    """
    if ner_model is None or ner_model.nlp is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    doc = ner_model.nlp(request.text)

    if request.use_rules and entity_rules is not None:
        rule_entities = entity_rules.extract_all(doc)
        doc = entity_rules.combine_with_statistical(
            doc, rule_entities, priority=request.priority
        )

    entities = []
    for ent in doc.ents:
        entities.append(EntityResult(
            text=ent.text,
            label=ent.label_,
            start=ent.start_char,
            end=ent.end_char,
            score=1.0,
        ))

    return PredictResponse(
        text=request.text,
        entities=entities,
        num_entities=len(entities),
    )


@app.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(request: BatchPredictRequest):
    """
    Extract entities from multiple texts in a single request.

    Uses spaCy's pipe for efficient batch processing.
    """
    if ner_model is None or ner_model.nlp is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    results = []
    total_entities = 0

    for doc in ner_model.nlp.pipe(request.texts, batch_size=64):
        if request.use_rules and entity_rules is not None:
            rule_entities = entity_rules.extract_all(doc)
            doc = entity_rules.combine_with_statistical(doc, rule_entities)

        entities = []
        for ent in doc.ents:
            entities.append(EntityResult(
                text=ent.text,
                label=ent.label_,
                start=ent.start_char,
                end=ent.end_char,
                score=1.0,
            ))

        total_entities += len(entities)
        results.append(PredictResponse(
            text=doc.text,
            entities=entities,
            num_entities=len(entities),
        ))

    return BatchPredictResponse(
        results=results,
        total_entities=total_entities,
    )


@app.post("/annotate")
async def annotate_text(request: AnnotateRequest):
    """
    Return annotated HTML with highlighted entity spans.

    Uses spaCy's displacy renderer to produce styled HTML output
    with color-coded entity labels.
    """
    if ner_model is None or ner_model.nlp is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    doc = ner_model.nlp(request.text)

    if entity_rules is not None:
        rule_entities = entity_rules.extract_all(doc)
        doc = entity_rules.combine_with_statistical(doc, rule_entities)

    options = {"compact": True}
    if request.colors:
        options["colors"] = request.colors
    else:
        options["colors"] = {
            "PERSON": "#aa9cfc",
            "ORGANIZATION": "#7aecec",
            "LOCATION": "#feca74",
            "PRODUCT": "#c887fb",
            "MONEY": "#e4e7d2",
            "DATE": "#bfe1d9",
        }

    html = displacy.render(doc, style="ent", options=options)
    return HTMLResponse(content=html)


@app.post("/train", response_model=TrainResponse)
async def train_model(request: TrainRequest):
    """
    Train the NER model on sample or provided data.

    For demonstration, generates synthetic training data and trains
    the model for the specified number of epochs.
    """
    if ner_model is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        preparator = SpaCyDataPreparator(config_path=CONFIG_PATH)

        if request.use_sample_data:
            training_data = preparator.generate_sample_data(num_samples=request.num_samples)
        else:
            raise HTTPException(
                status_code=400,
                detail="Custom training data upload not implemented in this endpoint. Use sample data.",
            )

        validation_report = preparator.validate_annotations(training_data)
        if validation_report["invalid_count"] > len(training_data) * 0.5:
            raise HTTPException(
                status_code=400,
                detail=f"Too many invalid annotations: {validation_report['invalid_count']}/{len(training_data)}",
            )

        train_set, dev_set = preparator.split_data(training_data, train_ratio=0.8)

        ner_model.create_blank_pipeline()
        history = ner_model.train(
            training_data=train_set,
            dev_data=dev_set,
            epochs=request.epochs,
        )

        os.makedirs("models/custom_ner", exist_ok=True)
        ner_model.save_model("models/custom_ner")

        global entity_rules
        entity_rules = EntityRules(ner_model.nlp, config_path=CONFIG_PATH)
        _setup_default_rules(entity_rules)

        return TrainResponse(
            status="success",
            epochs_trained=len(history["train_loss"]),
            best_f1=round(history.get("best_f1", 0.0), 4),
            labels=ner_model.labels,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Training failed: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@app.get("/labels", response_model=LabelsResponse)
async def get_labels():
    """Return the list of supported entity types and rule pattern summary."""
    if ner_model is None or ner_model.nlp is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    labels = list(ner_model.nlp.get_pipe("ner").labels) if "ner" in ner_model.nlp.pipe_names else []

    rule_summary = {}
    if entity_rules is not None:
        rule_summary = entity_rules.get_pattern_summary()

    return LabelsResponse(
        labels=labels,
        num_labels=len(labels),
        rule_patterns=rule_summary,
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Service health check endpoint."""
    if ner_model is None or ner_model.nlp is None:
        return HealthResponse(
            status="initializing",
            model_loaded=False,
            pipeline=[],
            num_labels=0,
        )

    return HealthResponse(
        status="healthy",
        model_loaded=True,
        pipeline=ner_model.nlp.pipe_names,
        num_labels=len(ner_model.labels),
    )

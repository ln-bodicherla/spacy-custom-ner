# Custom NER with SpaCy

Lightweight, production-ready Named Entity Recognition using SpaCy. Implements a custom training pipeline combining rule-based matching, statistical NER, and entity pattern matching for low-latency inference in production environments.

## Architecture

```
Input Text
    |
    v
SpaCy Pipeline
    |
    +---> Tokenizer
    |        |
    |        v
    +---> Entity Ruler (rule-based patterns)
    |        |
    |        v
    +---> Statistical NER Component (trained model)
    |        |
    |        v
    +---> Priority Resolution (merge rule + model predictions)
    |
    v
Extracted Entities [(text, label, start, end), ...]
```

## Comparison with Transformer-Based NER (Production Choice)

| Feature | SpaCy NER (this project) | Transformer NER (production) |
|---------|-------------------------|------------------------------|
| Architecture | CNN/transition-based | BERT/RoBERTa fine-tuned |
| Inference Latency | ~1-3ms per document | ~15-50ms per document |
| Training Data Needed | Moderate (500+ examples) | Low (few-shot capable) |
| Domain Adaptation | Requires substantial data | Better with limited data |
| Model Size | ~50-200 MB | ~400-1500 MB |
| Accuracy (F1) | 85-90% (domain-specific) | 90-95% (domain-specific) |
| Deployment | Simple (pip install) | GPU recommended |

**Decision:** Transformer-based NER was chosen for production to achieve higher accuracy on domain-specific entities with limited training data. SpaCy NER remains a strong choice for high-throughput scenarios where latency is critical and sufficient training data is available.

## Setup

### Prerequisites

- Python 3.10+
- SpaCy 3.7+

### Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

### Running

```bash
# Start API server
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

# Train custom model
python -m src.model --train --data data/training --config configs/config.yaml
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict` | Extract entities from text |
| POST | `/predict/batch` | Batch entity extraction |
| POST | `/annotate` | Returns annotated HTML |
| GET | `/labels` | List supported entity types |
| GET | `/health` | Service health check |

## Usage Example

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={"text": "Apple Inc. acquired Beats Electronics for $3 billion in May 2014."}
)
for entity in response.json()["entities"]:
    print(f"{entity['text']} -> {entity['label']} ({entity['score']:.2f})")
```

## Project Structure

```
spacy-custom-ner/
├── src/
│   ├── data_preparation.py    # Data loading and conversion
│   ├── model.py               # SpaCy NER model wrapper
│   ├── rules.py               # Rule-based entity patterns
│   └── api.py                 # FastAPI application
├── configs/
│   └── config.yaml            # Configuration
├── requirements.txt
└── README.md
```

## License

MIT License

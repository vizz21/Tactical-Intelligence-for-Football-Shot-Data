import json
import random

import pytest
from fastapi.testclient import TestClient

from football_rag import config as C
from football_rag.pipeline import Pipeline


@pytest.fixture(scope="session")
def pipeline():
    return Pipeline(device="cpu")


@pytest.fixture(scope="session")
def client():
    from app.main import app
    with TestClient(app) as c:          # runs the lifespan (loads the pipeline once)
        yield c


@pytest.fixture(scope="session")
def val_template_queries():
    """A fixed random sample of 30 unique val template queries with their attrs."""
    seen = {}
    with open(C.PAIRS, encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            if p["split"] == "val":
                seen.setdefault(p["query"], p["attrs"])
    items = sorted(seen.items())
    return random.Random(0).sample(items, 30)

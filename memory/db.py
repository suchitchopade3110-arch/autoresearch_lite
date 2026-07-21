import os
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any, Optional
import uuid
import json

class ExperimentDB:
    def __init__(self, db_path: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=db_path)
        # using default sentence-transformers model embedded in chromadb
        self.ef = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name="experiments",
            embedding_function=self.ef
        )

    def store_experiment(
        self,
        hypothesis: str,
        diff: str,
        rationale: str,
        metrics: Dict[str, float],
        outcome: str,
        failure_reason: Optional[str] = None
    ) -> str:
        """Stores an experiment run in the vector DB."""
        record_id = uuid.uuid4().hex

        # Combine text for embedding so it's retrievable by similar hypotheses or failure contexts
        document = f"Hypothesis: {hypothesis}\nRationale: {rationale}\nOutcome: {outcome}"
        if failure_reason:
            document += f"\nFailure Reason: {failure_reason}"

        metadata = {
            "hypothesis": hypothesis,
            "diff": diff,
            "rationale": rationale,
            "metrics": json.dumps(metrics),
            "outcome": outcome,
            "failure_reason": failure_reason or ""
        }

        self.collection.add(
            documents=[document],
            metadatas=[metadata],
            ids=[record_id]
        )
        return record_id

    def retrieve_experiments(self, query: str, k: int = 3, filter_outcome: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieves top-k most similar experiments based on query text."""
        where_clause = {}
        if filter_outcome:
            where_clause = {"outcome": filter_outcome}

        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            where=where_clause if where_clause else None
        )

        retrieved = []
        if results and results['metadatas'] and len(results['metadatas']) > 0:
            for metadata in results['metadatas'][0]:
                meta = dict(metadata)
                meta['metrics'] = json.loads(meta['metrics'])
                retrieved.append(meta)

        return retrieved

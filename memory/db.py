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
        # cosine distance is bounded and near 0 for near-identical text, which
        # is what duplicate_checker.py's small default threshold (0.1) assumes -
        # chromadb's collection default (L2) doesn't range near 0 for this kind
        # of text, so that threshold would never trigger on real near-duplicates.
        self.collection = self.client.get_or_create_collection(
            name="experiments",
            embedding_function=self.ef,
            metadata={"hnsw:space": "cosine"}
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

        # Combine text for embedding so it's retrievable by similar hypotheses,
        # failure contexts, OR similar diffs - duplicate_checker.py queries by
        # diff text specifically, so the diff itself must be part of what's
        # embedded, not just stored as unsearched metadata.
        document = f"Hypothesis: {hypothesis}\nRationale: {rationale}\nOutcome: {outcome}\nDiff:\n{diff}"
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
            distances = results['distances'][0] if results.get('distances') else []
            for i, metadata in enumerate(results['metadatas'][0]):
                meta = dict(metadata)
                meta['metrics'] = json.loads(meta['metrics'])
                if i < len(distances):
                    meta['distance'] = distances[i]
                retrieved.append(meta)

        return retrieved

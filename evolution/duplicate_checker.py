from memory.db import ExperimentDB

def is_duplicate(diff: str, db: ExperimentDB, threshold: float = 0.1) -> bool:
    if not diff or diff.strip() == "":
        return False

    results = db.retrieve_experiments(query=diff, k=1)

    if not results:
        return False

    closest = results[0]
    distance = closest.get('distance', float('inf'))

    if diff.strip() == closest['diff'].strip():
        return True

    if distance < threshold:
        return True

    return False
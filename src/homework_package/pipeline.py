from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def create_pipeline(
    use_scaler: bool, max_iter: int, logreg_C: float, random_state: int
) -> Pipeline:
    pipeline_steps = []
    if use_scaler:
        pipeline_steps.append(("scaler", StandardScaler()))
    pipeline_steps.append(
        (
            "classifier",
            KMeans(
                random_state=random_state, max_iter=max_iter
            ),
        )
    )
    return Pipeline(steps=pipeline_steps)
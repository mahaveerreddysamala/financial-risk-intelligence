# Model Run Registry

Phase 10 adds a lightweight model-run registry for reproducible experiment metadata.

Each run record stores:

- deterministic run ID derived from model name, parameters, and metrics
- model name
- training/evaluation parameters
- measured evaluation metrics
- feature count
- model artifact path

The registry writes JSON metadata records to an output directory so experiments can be reviewed or promoted without coupling the portfolio implementation to an external tracking service.

## Production path

The same metadata contract can later be emitted to MLflow, SageMaker Model Registry, or another managed registry. The artifact path is intentionally stored separately from metrics so model binaries can live in object storage while metadata remains queryable.

This registry does not claim to replace MLflow or a managed model registry; it demonstrates the underlying experiment-tracking contract in a dependency-light form.

# Synthetic investigation reference — demonstration guidance, not legal policy

## Shared devices and network reuse
Shared devices across accounts can indicate a relationship worth investigating. Device reuse alone is not proof of fraud: families and shared terminals can produce legitimate overlap. Review transaction evidence and connected entities before drawing conclusions.

## Transaction velocity
Transaction velocity measures recent activity within a defined time window. An unusual burst can merit review, but legitimate purchases can also cluster. Historical features must exclude the transaction currently being scored and future events.

## Investigation capacity
Review capacity determines how many transactions enter the ranked investigation queue. Precision measures the fraction of reviewed transactions labeled fraud; recall measures the fraction of all fraud captured. Lift compares queue precision with the evaluated population's fraud prevalence. These are retrospective synthetic evaluation metrics, not guarantees.

## Model and anomaly signals
The dashboard combines fraud model, anomaly, network and velocity signals into a bounded risk score. A high score is a prioritization signal, not a determination of criminal intent. Analysts should inspect individual signals and case evidence. Synthetic labels are evaluation-only and must not become scoring inputs.

## Human review and limitations
The copilot is read-only and cannot block payments or resolve investigation cases. It retrieves reference excerpts and may optionally draft an LLM response for human review. Source-ID validation does not establish that a generated statement is supported. Do not submit personal data, credentials, or real financial records.

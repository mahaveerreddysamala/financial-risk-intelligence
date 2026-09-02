# Grounded Investigation Copilot

Phase 12 adds a retrieval and prompt-construction layer for a future LLM-powered financial investigation assistant.

## Architecture

```text
Investigation Case
      |
      +--> Structured evidence
      |
      v
Policy / Fraud Typology Documents
      |
      v
TF-IDF Retrieval
      |
      +------------------+
      |                  |
      v                  v
Case Evidence      Retrieved References
      |                  |
      +--------+---------+
               v
        Grounded Prompt
               |
               v
        LLM / GenAI Adapter
```

The current implementation deliberately does not require an external LLM provider. It establishes the retrieval, evidence, and grounding contract first so a production adapter can later connect to an approved model service.

## Grounding Boundary

The generated prompt explicitly requires the downstream model to use only supplied case evidence and retrieved reference documents. It must distinguish observed evidence from interpretation and state when evidence is insufficient.

This prevents the portfolio implementation from presenting unsupported generated claims as transaction facts or policy requirements.

## Retrieval

Reference documents contain `document_id` and `text`. TF-IDF bigram retrieval provides a dependency-light baseline for policy and fraud-typology lookup. A production deployment could replace this component with an embedding model and vector store without changing the case/evidence contract.

## Production Extension

A later adapter can map `CopilotContext` into an LLM request and return a structured response containing a summary, evidence-linked findings, references, and analyst next steps. The structured evidence remains the source of truth.

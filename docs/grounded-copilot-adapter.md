# Grounded Copilot Adapter

Phase 12 adds a provider-neutral adapter between structured investigation evidence and an external large language model.

## Design

`CopilotContext` is converted into the existing grounded prompt contract, then passed to a small `TextGenerator` protocol. This keeps provider-specific SDKs outside the core investigation package.

The adapter requires a non-empty generated response and returns a `CopilotResponse` that preserves the case ID, exact grounded prompt, and response text.

A deterministic evidence-only fallback is provided for environments where no external LLM is configured.

## Grounding Boundary

The adapter does not add facts or independently retrieve evidence. The prompt instructs the downstream model to use only supplied case evidence and retrieved reference documents and to disclose insufficiency rather than inventing facts.

The fallback likewise emits only supplied evidence fields and reference identifiers.

## Production Path

A production implementation can provide the `TextGenerator` protocol using an approved hosted model or enterprise gateway. Authentication, data-loss prevention, prompt/response logging, model access controls, and retention policies should remain in the application infrastructure rather than inside the domain library.

# Security

This repository is a portfolio implementation and uses synthetic transaction data for demonstrations and tests.

## Reporting

Do not open a public issue containing credentials, API keys, tokens, private transaction data, or other sensitive information. Report suspected security issues privately through the repository owner's GitHub contact channel.

## Deployment security expectations

Production deployments should add:

- managed secret storage and runtime secret injection
- IAM least-privilege roles
- private networking and restricted ingress
- TLS for API, Kafka, Redis, and service-to-service communication
- encryption at rest and in transit
- centralized audit logging and alerting
- dependency and container vulnerability scanning
- model artifact integrity and access controls

The local Docker Compose environment intentionally favors reproducible development and is not a production security boundary.

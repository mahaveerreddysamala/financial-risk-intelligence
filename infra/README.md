# Infrastructure

This directory contains a cloud-ready infrastructure foundation for the Financial Risk Intelligence platform.

## Layout

```text
infra/
├── README.md
└── terraform/
    └── aws/
        ├── main.tf
        ├── outputs.tf
        ├── variables.tf
        └── versions.tf
```

## AWS foundation

The Terraform stack creates only foundational, low-risk resources:

- ECR repository for the application image
- ECS cluster for container workloads
- S3 bucket for model/data artifacts
- CloudWatch log group for centralized container logs

It does **not** automatically create a production VPC, load balancer, Kafka cluster, Redis cluster, database, certificates, secrets, or public ingress. Those resources are environment-specific and should be composed with the organization's networking and security modules.

## Usage

```bash
cd infra/terraform/aws
terraform init
terraform fmt -check
terraform validate
terraform plan
```

Set the bucket suffix and AWS region through Terraform variables or a `.tfvars` file that is kept outside source control.

Never commit cloud credentials, secret values, `.tfstate`, or environment-specific `.tfvars` files.

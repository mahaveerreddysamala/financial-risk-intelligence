variable "aws_region" {
  description = "AWS region for the foundation resources."
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Deployment environment name."
  type        = string
  default     = "portfolio"
}

variable "artifact_bucket_suffix" {
  description = "Globally unique suffix for the S3 artifact bucket."
  type        = string

  validation {
    condition     = length(var.artifact_bucket_suffix) >= 6 && can(regex("^[a-z0-9-]+$", var.artifact_bucket_suffix))
    error_message = "artifact_bucket_suffix must be at least 6 characters and contain only lowercase letters, digits, and hyphens."
  }
}

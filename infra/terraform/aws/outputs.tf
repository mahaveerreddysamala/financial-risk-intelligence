output "ecr_repository_url" {
  description = "ECR repository URL for the application image."
  value       = aws_ecr_repository.app.repository_url
}

output "ecs_cluster_name" {
  description = "ECS cluster name for application services."
  value       = aws_ecs_cluster.app.name
}

output "artifact_bucket_name" {
  description = "Private S3 bucket for model and data artifacts."
  value       = aws_s3_bucket.artifacts.bucket
}

output "cloudwatch_log_group" {
  description = "CloudWatch log group for application workloads."
  value       = aws_cloudwatch_log_group.app.name
}

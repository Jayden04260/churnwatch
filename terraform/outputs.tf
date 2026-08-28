output "function_url" {
  description = "Public HTTPS endpoint for the API (append /health or /predict)."
  value       = aws_lambda_function_url.api.function_url
}

output "ecr_repository_url" {
  value = aws_ecr_repository.api.repository_url
}

output "data_bucket" {
  value = aws_s3_bucket.data.bucket
}

output "lambda_function_arn" {
  value = aws_lambda_function.api.arn
}

output "drift_check_ecr_repository_url" {
  value = aws_ecr_repository.drift_check.repository_url
}

output "drift_alerts_topic_arn" {
  value = aws_sns_topic.drift_alerts.arn
}

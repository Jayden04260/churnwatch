variable "aws_region" {
  description = "AWS region - same account/region as emotisense for consistency."
  type        = string
  default     = "eu-north-1"
}

variable "function_name" {
  description = "Name shared by the Lambda function and ECR repository."
  type        = string
  default     = "churnwatch-api"
}

variable "image_uri" {
  description = <<-EOT
    Full ECR image URI (digest-pinned, not `:latest`) for the API Lambda
    (../Dockerfile). Terraform manages the function's infrastructure, not
    the image itself - build and push it separately, then pass the
    resulting digest here.
  EOT
  type        = string
  default     = ""
}

variable "drift_check_image_uri" {
  description = "Same as image_uri, but for the drift_check Lambda (../Dockerfile.drift_check)."
  type        = string
  default     = ""
}

variable "drift_check_schedule" {
  description = <<-EOT
    EventBridge schedule expression for the drift-check job. Started at
    every 10 minutes for demo purposes (so drift was actually observable
    within a reasonable demo/interview timeframe) - now that it's proven
    itself (a real alert email received 2026-09-01), dialed back to once a
    day, matching a realistic production cadence for a model that doesn't
    meaningfully drift minute-to-minute.
  EOT
  type        = string
  default     = "rate(1 day)"
}

variable "memory_size" {
  description = <<-EOT
    Lambda memory in MB. Much lower than emotisense's 3008MB - this is a
    small scikit-learn model baked straight into the image with no
    Hugging-Face-Hub-style download/load step, so there's no cold-start
    memory spike to buy headroom against.
  EOT
  type        = number
  default     = 512
}

variable "timeout_seconds" {
  type    = number
  default = 30
}

variable "deploy_user_name" {
  description = <<-EOT
    IAM user this stack grants scoped deploy permissions to. Deliberately
    a separate user from emotisense-deploy, not a broadened version of it -
    each project's deploy automation gets its own least-privilege identity
    rather than one user's blast radius growing to cover every project.
  EOT
  type        = string
  default     = "churnwatch-deploy"
}

variable "alert_email" {
  description = "Email address for drift-alert SNS notifications (Phase C)."
  type        = string
  default     = "jdsteadman04@gmail.com"
}

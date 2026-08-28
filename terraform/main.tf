data "aws_caller_identity" "current" {}

# ---------------------------------------------------------------------------
# ECR - holds the Docker image (see ../Dockerfile).
# ---------------------------------------------------------------------------
resource "aws_ecr_repository" "api" {
  name = var.function_name

  image_scanning_configuration {
    scan_on_push = true
  }
}

# ---------------------------------------------------------------------------
# Lambda execution role. Unlike emotisense's (CloudWatch Logs only), this
# one also needs S3 read/write for prediction logging + drift baseline
# reads (Phase C) - added here now rather than bolted on later, since the
# role itself doesn't change shape between phases, only its attached
# policy's actions do.
# ---------------------------------------------------------------------------
data "aws_iam_policy_document" "lambda_assume_role" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "lambda_exec" {
  name               = "${var.function_name}-lambda-role"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume_role.json
}

resource "aws_iam_role_policy_attachment" "lambda_basic_execution" {
  role       = aws_iam_role.lambda_exec.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

data "aws_iam_policy_document" "lambda_s3_access" {
  statement {
    sid    = "PredictionLogsAndModels"
    effect = "Allow"
    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:ListBucket",
    ]
    resources = [
      aws_s3_bucket.data.arn,
      "${aws_s3_bucket.data.arn}/*",
    ]
  }
}

resource "aws_iam_role_policy" "lambda_s3_access" {
  name   = "s3-access"
  role   = aws_iam_role.lambda_exec.id
  policy = data.aws_iam_policy_document.lambda_s3_access.json
}

# ---------------------------------------------------------------------------
# S3 bucket for prediction logs, model versions, and drift reports (Phase C
# writes to logs/ and drift_reports/, train.py/retrain.py write to models/).
# Bucket name must be globally unique, hence the account ID suffix.
# ---------------------------------------------------------------------------
resource "aws_s3_bucket" "data" {
  bucket = "${var.function_name}-${data.aws_caller_identity.current.account_id}"
}

# ---------------------------------------------------------------------------
# The API Lambda function.
# ---------------------------------------------------------------------------
resource "aws_lambda_function" "api" {
  function_name = var.function_name
  role          = aws_iam_role.lambda_exec.arn

  package_type = "Image"
  image_uri    = var.image_uri

  memory_size = var.memory_size
  timeout     = var.timeout_seconds

  environment {
    variables = {
      DATA_BUCKET = aws_s3_bucket.data.bucket
    }
  }

  depends_on = [aws_iam_role_policy_attachment.lambda_basic_execution]
}

# ---------------------------------------------------------------------------
# Public Function URL - same two-permission-statement pattern as emotisense
# (see that project's main.tf for the full writeup of why the second
# statement needs a local-exec workaround).
# ---------------------------------------------------------------------------
resource "aws_lambda_function_url" "api" {
  function_name      = aws_lambda_function.api.function_name
  authorization_type = "NONE"
  invoke_mode        = "BUFFERED"
}

resource "aws_lambda_permission" "url_invoke_function_url" {
  statement_id           = "FunctionURLAllowPublicAccess"
  action                 = "lambda:InvokeFunctionUrl"
  function_name          = aws_lambda_function.api.function_name
  principal              = "*"
  function_url_auth_type = "NONE"
}

# aws_lambda_permission has no argument for the lambda:InvokedViaFunctionUrl
# condition (confirmed via `terraform providers schema -json` on emotisense,
# same provider version) - shelling out to the CLI for just this one
# statement, single-line command (Windows local-exec runs via cmd.exe).
resource "terraform_data" "url_invoke_function_permission" {
  triggers_replace = [aws_lambda_function.api.function_name, var.aws_region]

  provisioner "local-exec" {
    command = "aws lambda add-permission --function-name ${self.triggers_replace[0]} --statement-id UrlPolicyInvokeFunction --action lambda:InvokeFunction --principal \"*\" --invoked-via-function-url --region ${self.triggers_replace[1]} || exit 0"
  }

  provisioner "local-exec" {
    when    = destroy
    command = "aws lambda remove-permission --function-name ${self.triggers_replace[0]} --statement-id UrlPolicyInvokeFunction --region ${self.triggers_replace[1]} || exit 0"
  }
}

# ---------------------------------------------------------------------------
# PHASE C - drift detection, alerting.
# ---------------------------------------------------------------------------

resource "aws_sns_topic" "drift_alerts" {
  name = "${var.function_name}-drift-alerts"
}

resource "aws_sns_topic_subscription" "drift_alerts_email" {
  topic_arn = aws_sns_topic.drift_alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email
}

data "aws_iam_policy_document" "lambda_sns_publish" {
  statement {
    sid       = "PublishDriftAlerts"
    effect    = "Allow"
    actions   = ["sns:Publish"]
    resources = [aws_sns_topic.drift_alerts.arn]
  }
}

resource "aws_iam_role_policy" "lambda_sns_publish" {
  name   = "sns-publish"
  role   = aws_iam_role.lambda_exec.id
  policy = data.aws_iam_policy_document.lambda_sns_publish.json
}

resource "aws_ecr_repository" "drift_check" {
  name = "${var.function_name}-drift-check"

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_lambda_function" "drift_check" {
  function_name = "${var.function_name}-drift-check"
  role          = aws_iam_role.lambda_exec.arn

  package_type = "Image"
  image_uri    = var.drift_check_image_uri

  # More headroom than the API function: this loads the day's full
  # prediction log (pandas) rather than scoring one request at a time.
  memory_size = 512
  timeout     = 60

  environment {
    variables = {
      DATA_BUCKET    = aws_s3_bucket.data.bucket
      SNS_TOPIC_ARN  = aws_sns_topic.drift_alerts.arn
    }
  }

  depends_on = [aws_iam_role_policy_attachment.lambda_basic_execution]
}

resource "aws_cloudwatch_event_rule" "drift_check_schedule" {
  name                = "${var.function_name}-drift-check-schedule"
  schedule_expression = var.drift_check_schedule
}

resource "aws_cloudwatch_event_target" "drift_check" {
  rule = aws_cloudwatch_event_rule.drift_check_schedule.name
  arn  = aws_lambda_function.drift_check.arn
}

resource "aws_lambda_permission" "allow_eventbridge" {
  statement_id  = "AllowEventBridgeInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.drift_check.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.drift_check_schedule.arn
}

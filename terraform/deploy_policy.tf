# Scoped permissions for the IAM user that deploys this stack
# (churnwatch-deploy) - a separate identity from emotisense-deploy, not a
# widened version of it. Same discipline as emotisense: this needs to be
# bootstrapped once with a temporarily-attached AdministratorAccess policy,
# then verified that this scoped policy alone is sufficient (by actually
# detaching admin and re-running `terraform plan`) before treating it as
# final - see emotisense/terraform/deploy_policy.tf for the full writeup of
# why that empirical verification step matters more than writing a policy
# that merely looks minimal.

data "aws_iam_policy_document" "deploy" {
  statement {
    sid       = "EcrAuth"
    effect    = "Allow"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }

  statement {
    sid    = "EcrPushPull"
    effect = "Allow"
    actions = [
      "ecr:CreateRepository",
      "ecr:DescribeRepositories",
      "ecr:DescribeImages",
      "ecr:ListImages",
      "ecr:ListTagsForResource",
      "ecr:BatchCheckLayerAvailability",
      "ecr:BatchGetImage",
      "ecr:GetDownloadUrlForLayer",
      "ecr:InitiateLayerUpload",
      "ecr:UploadLayerPart",
      "ecr:CompleteLayerUpload",
      "ecr:PutImage",
    ]
    resources = [aws_ecr_repository.api.arn, aws_ecr_repository.drift_check.arn]
  }

  statement {
    sid    = "LambdaManageFunction"
    effect = "Allow"
    actions = [
      "lambda:CreateFunction",
      "lambda:GetFunction",
      "lambda:GetFunctionConfiguration",
      "lambda:ListVersionsByFunction",
      "lambda:UpdateFunctionCode",
      "lambda:UpdateFunctionConfiguration",
      "lambda:DeleteFunction",
      "lambda:TagResource",
      "lambda:ListTags",
      "lambda:CreateFunctionUrlConfig",
      "lambda:GetFunctionUrlConfig",
      "lambda:UpdateFunctionUrlConfig",
      "lambda:DeleteFunctionUrlConfig",
      "lambda:AddPermission",
      "lambda:RemovePermission",
      "lambda:GetPolicy",
    ]
    resources = [aws_lambda_function.api.arn, aws_lambda_function.drift_check.arn]
  }

  statement {
    sid    = "ManageLambdaExecRole"
    effect = "Allow"
    actions = [
      "iam:CreateRole",
      "iam:GetRole",
      "iam:DeleteRole",
      "iam:AttachRolePolicy",
      "iam:DetachRolePolicy",
      "iam:PutRolePolicy",
      "iam:DeleteRolePolicy",
      "iam:ListAttachedRolePolicies",
      "iam:ListRolePolicies",
      "iam:GetRolePolicy",
      "iam:TagRole",
    ]
    resources = [aws_iam_role.lambda_exec.arn]
  }

  statement {
    sid       = "PassLambdaExecRole"
    effect    = "Allow"
    actions   = ["iam:PassRole"]
    resources = [aws_iam_role.lambda_exec.arn]

    condition {
      test     = "StringEquals"
      variable = "iam:PassedToService"
      values   = ["lambda.amazonaws.com"]
    }
  }

  statement {
    sid    = "ManageDataBucket"
    effect = "Allow"
    actions = [
      "s3:CreateBucket",
      "s3:GetBucketLocation",
      "s3:ListBucket",
      "s3:GetObject",
      "s3:PutObject",
      "s3:DeleteObject",
    ]
    resources = [
      aws_s3_bucket.data.arn,
      "${aws_s3_bucket.data.arn}/*",
    ]
  }

  # Self-referential (see emotisense's deploy_policy.tf for why): this
  # policy and its attachment to the deploy user are themselves
  # Terraform-managed resources.
  statement {
    sid    = "ManageOwnDeployPolicy"
    effect = "Allow"
    actions = [
      "iam:GetPolicy",
      "iam:GetPolicyVersion",
      "iam:ListPolicyVersions",
      "iam:CreatePolicyVersion",
      "iam:DeletePolicyVersion",
      "iam:TagPolicy",
    ]
    resources = ["arn:aws:iam::${data.aws_caller_identity.current.account_id}:policy/${var.function_name}-deploy-policy"]
  }

  statement {
    sid    = "ManageOwnUserPolicyAttachment"
    effect = "Allow"
    actions = [
      "iam:ListAttachedUserPolicies",
      "iam:AttachUserPolicy",
      "iam:DetachUserPolicy",
    ]
    resources = ["arn:aws:iam::${data.aws_caller_identity.current.account_id}:user/${var.deploy_user_name}"]
  }

  statement {
    sid    = "ReadLambdaLogs"
    effect = "Allow"
    actions = [
      "logs:DescribeLogGroups",
      "logs:DescribeLogStreams",
      "logs:FilterLogEvents",
      "logs:GetLogEvents",
    ]
    resources = [
      "arn:aws:logs:${var.aws_region}:${data.aws_caller_identity.current.account_id}:log-group:/aws/lambda/${var.function_name}",
      "arn:aws:logs:${var.aws_region}:${data.aws_caller_identity.current.account_id}:log-group:/aws/lambda/${var.function_name}:*",
      "arn:aws:logs:${var.aws_region}:${data.aws_caller_identity.current.account_id}:log-group:/aws/lambda/${var.function_name}-drift-check",
      "arn:aws:logs:${var.aws_region}:${data.aws_caller_identity.current.account_id}:log-group:/aws/lambda/${var.function_name}-drift-check:*",
    ]
  }

  statement {
    sid    = "ManageDriftSchedule"
    effect = "Allow"
    actions = [
      "events:PutRule",
      "events:DescribeRule",
      "events:PutTargets",
      "events:RemoveTargets",
      "events:DeleteRule",
      "events:ListTargetsByRule",
    ]
    resources = [aws_cloudwatch_event_rule.drift_check_schedule.arn]
  }

  statement {
    sid    = "ManageDriftAlertsTopic"
    effect = "Allow"
    actions = [
      "sns:CreateTopic",
      "sns:GetTopicAttributes",
      "sns:SetTopicAttributes",
      "sns:DeleteTopic",
      "sns:Subscribe",
      "sns:Unsubscribe",
      "sns:ListSubscriptionsByTopic",
      "sns:TagResource",
    ]
    resources = [aws_sns_topic.drift_alerts.arn]
  }
}

resource "aws_iam_policy" "deploy" {
  name        = "${var.function_name}-deploy-policy"
  description = "Scoped deploy permissions for ${var.function_name} - ECR push (both images), the two Lambdas + their shared role, the S3 data bucket, the drift-check EventBridge schedule, the SNS alerts topic, log read access. Nothing else."
  policy      = data.aws_iam_policy_document.deploy.json
}

resource "aws_iam_user_policy_attachment" "deploy" {
  user       = var.deploy_user_name
  policy_arn = aws_iam_policy.deploy.arn
}

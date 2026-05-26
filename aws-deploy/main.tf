# AWS Terraform Configuration for AgentEcon Fabric WAN Testbed
# 4 validator peers + 1 orderer in us-east-1

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

variable "aws_region" {
  description = "AWS region for deployment"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Project name for resource naming"
  type        = string
  default     = "agentecon"
}

variable "vpc_cidr" {
  description = "VPC CIDR block"
  type        = string
  default     = "10.0.0.0/16"
}

variable "instance_type" {
  description = "EC2 instance type for validator peers"
  type        = string
  default     = "c6i.xlarge"
}

variable "orderer_instance_type" {
  description = "EC2 instance type for orderer"
  type        = string
  default     = "t3.medium"
}

variable "logical_validators_per_peer" {
  description = "Logical validator client identities driven through each physical Fabric peer for the 64-validator benchmark"
  type        = number
  default     = 16
}

variable "key_name" {
  description = "SSH key pair name"
  type        = string
  default     = "agentecon-key"
}

variable "allowed_cidr" {
  description = "CIDR block allowed to access instances"
  type        = string
  default     = "0.0.0.0/0"
}

# VPC
resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "${var.project_name}-vpc"
  }
}

# Internet Gateway
resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id

  tags = {
    Name = "${var.project_name}-igw"
  }
}

# Subnet for validators
resource "aws_subnet" "validators" {
  vpc_id                  = aws_vpc.main.id
  cidr_block              = "10.0.1.0/24"
  availability_zone       = "${var.aws_region}a"
  map_public_ip_on_launch = true

  tags = {
    Name = "${var.project_name}-validators-subnet"
  }
}

# Subnet for orderer
resource "aws_subnet" "orderer" {
  vpc_id                  = aws_vpc.main.id
  cidr_block              = "10.0.2.0/24"
  availability_zone       = "${var.aws_region}b"
  map_public_ip_on_launch = true

  tags = {
    Name = "${var.project_name}-orderer-subnet"
  }
}

# Route Table
resource "aws_route_table" "main" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }

  tags = {
    Name = "${var.project_name}-rt"
  }
}

resource "aws_route_table_association" "validators" {
  subnet_id      = aws_subnet.validators.id
  route_table_id = aws_route_table.main.id
}

resource "aws_route_table_association" "orderer" {
  subnet_id      = aws_subnet.orderer.id
  route_table_id = aws_route_table.main.id
}

# Security Group
resource "aws_security_group" "fabric" {
  name        = "${var.project_name}-fabric-sg"
  description = "Security group for Fabric network"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.allowed_cidr]
    description = "SSH"
  }

  ingress {
    from_port   = 7050
    to_port     = 8050
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/16"]
    description = "Fabric Peer/Orderer"
  }

  ingress {
    from_port   = 8443
    to_port     = 8443
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/16"]
    description = "Fabric Operations"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.project_name}-sg"
  }
}

# IAM Role for EC2
resource "aws_iam_role" "ec2_role" {
  name = "${var.project_name}-ec2-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "ec2.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_instance_profile" "ec2_profile" {
  name = "${var.project_name}-ec2-profile"
  role = aws_iam_role.ec2_role.name
}

# S3 Bucket for ledger snapshots
resource "aws_s3_bucket" "ledger_snapshots" {
  bucket = "${var.project_name}-ledger-${data.aws_caller_identity.current.account_id}"

  tags = {
    Name        = "${var.project_name}-ledger"
    Project     = var.project_name
    Environment = "production"
  }
}

resource "aws_s3_bucket_versioning" "ledger_snapshots" {
  bucket = aws_s3_bucket.ledger_snapshots.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "ledger_snapshots" {
  bucket = aws_s3_bucket.ledger_snapshots.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# EC2 Instance for validator peers
resource "aws_instance" "validator" {
  count                = 4
  ami                  = data.aws_ami.ubuntu.id
  instance_type        = var.instance_type
  subnet_id            = aws_subnet.validators.id
  key_name             = var.key_name
  iam_instance_profile = aws_iam_instance_profile.ec2_profile.name

  vpc_security_group_ids = [aws_security_group.fabric.id]

  user_data = templatefile("${path.module}/templates/validator-init.tpl", {
    peer_index                  = count.index
    project_name                = var.project_name
    logical_validators_per_peer = var.logical_validators_per_peer
  })

  root_block_device {
    volume_size = 100
    volume_type = "gp3"
    encrypted   = true
  }

  tags = {
    Name  = "${var.project_name}-validator-${count.index}"
    Role  = "validator"
    Index = count.index
  }
}

# EC2 Instance for orderer
resource "aws_instance" "orderer" {
  ami                  = data.aws_ami.ubuntu.id
  instance_type        = var.orderer_instance_type
  subnet_id            = aws_subnet.orderer.id
  key_name             = var.key_name
  iam_instance_profile = aws_iam_instance_profile.ec2_profile.name

  vpc_security_group_ids = [aws_security_group.fabric.id]

  user_data = templatefile("${path.module}/templates/orderer-init.tpl", {
    project_name = var.project_name
  })

  root_block_device {
    volume_size = 50
    volume_type = "gp3"
    encrypted   = true
  }

  tags = {
    Name = "${var.project_name}-orderer"
    Role = "orderer"
  }
}

# CloudWatch Dashboard
resource "aws_cloudwatch_dashboard" "main" {
  dashboard_name = "${var.project_name}-dashboard"

  dashboard_body = jsonencode({
    widgets = [
      {
        type = "metric"
        properties = {
          metrics = [
            ["AWS/EC2", "CPUUtilization", { stat = "Average" }],
            [".", "NetworkIn", { stat = "Sum" }],
            [".", "NetworkOut", { stat = "Sum" }]
          ]
          period = 300
          stat   = "Average"
          region = var.aws_region
          title  = "EC2 Metrics"
        }
      }
    ]
  })
}

# Budget Alert
resource "aws_budgets_budget" "monthly_cost" {
  name              = "${var.project_name}-monthly-budget"
  budget_type       = "COST"
  limit_amount      = "150"
  limit_unit        = "USD"
  time_period_start = "2024-01-01_00:00"
  time_unit         = "MONTHLY"
}

data "aws_caller_identity" "current" {}

data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

output "vpc_id" {
  value = aws_vpc.main.id
}

output "validator_ips" {
  description = "IP addresses of validator peers"
  value       = aws_instance.validator[*].public_ip
}

output "logical_validator_count" {
  description = "Logical validator identities represented by the physical peers"
  value       = length(aws_instance.validator) * var.logical_validators_per_peer
}

output "orderer_ip" {
  description = "IP address of orderer"
  value       = aws_instance.orderer.public_ip
}

output "s3_bucket" {
  description = "S3 bucket for ledger snapshots"
  value       = aws_s3_bucket.ledger_snapshots.bucket
}

output "logical_validator_capacity" {
  description = "Total logical validator clients represented by the 4 physical peers"
  value       = length(aws_instance.validator) * var.logical_validators_per_peer
}

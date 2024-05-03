terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "5.35.0"
    }
  }

}

locals {
  instance_type            = "g4dn.2xlarge"
  ami_id                   = "ami-0f838ce49da78a525" # https://ap-southeast-1.console.aws.amazon.com/ec2/home?region=ap-southeast-1#Images:visibility=owned-by-me
  aws_iam_instance_profile = "solana_is_good"
  user_name                = "{{bucket_name}}"
  key_name                 = "xuananle"
  project_name             = "{{project_name}}"
  region                   = "ap-southeast-1"
}
provider "aws" {
  region = local.region
}

data "aws_iam_instance_profile" "existing_role" {
  name = local.aws_iam_instance_profile
}

resource "aws_instance" "example" {
  ami                  = local.ami_id
  instance_type        = local.instance_type
  key_name             = local.key_name
  iam_instance_profile = data.aws_iam_instance_profile.existing_role.role_name
  user_data = templatefile("user_data.sh.tpl", {
    bucket_name  = local.user_name,
    project_name = local.project_name
    }
  )
  instance_initiated_shutdown_behavior = "terminate"
  ebs_block_device {
    # 30 gb ssd
    device_name = "/dev/xvda"
    volume_size = 100
    volume_type = "gp3"
  }

  tags = {
    Name = "${local.user_name}-${local.project_name}"
  }
}
output "ec2_instance_info" {
  description = "Information about the EC2 instance"
  value = {
    public_ip = aws_instance.example.public_ip
    tags      = aws_instance.example.tags.Name
  }
} 


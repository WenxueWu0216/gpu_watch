# Two GPU nodes in one cluster placement group for GPUWatch live validation.
# NOT VERIFIED IN CI: CI runs `terraform validate` only; nothing is applied.

data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
  filter {
    name   = "default-for-az"
    values = ["true"]
  }
}

data "aws_ami" "dlami" {
  count       = var.ami_id == "" ? 1 : 0
  most_recent = true
  owners      = ["amazon"]
  filter {
    name   = "name"
    values = ["Deep Learning OSS Nvidia Driver AMI GPU PyTorch * (Ubuntu 22.04) *"]
  }
  filter {
    name   = "architecture"
    values = ["x86_64"]
  }
}

locals {
  ami_id = var.ami_id != "" ? var.ami_id : data.aws_ami.dlami[0].id
  tags   = { Project = "gpuwatch", ManagedBy = "terraform" }
}

resource "aws_placement_group" "cluster" {
  name     = "${var.name}-pg"
  strategy = "cluster"
  tags     = local.tags
}

resource "aws_security_group" "nodes" {
  name        = "${var.name}-nodes"
  description = "GPUWatch nodes: SSH from operator, all traffic between nodes"
  vpc_id      = data.aws_vpc.default.id
  tags        = local.tags

  ingress {
    description = "SSH from operator"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.ssh_cidr]
  }

  ingress {
    description = "All traffic between cluster nodes (NCCL, torchrun rendezvous, Prometheus scrape)"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    self        = true
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_instance" "gpu" {
  count                  = var.instance_count
  ami                    = local.ami_id
  instance_type          = var.instance_type
  key_name               = var.key_name
  subnet_id              = data.aws_subnets.default.ids[0]
  placement_group        = aws_placement_group.cluster.id
  vpc_security_group_ids = [aws_security_group.nodes.id]

  root_block_device {
    volume_size = var.root_volume_gb
    volume_type = "gp3"
    encrypted   = true
  }

  metadata_options {
    http_tokens = "required"
  }

  user_data = templatefile("${path.module}/cloud-init.sh.tftpl", {
    repo_url = var.repo_url
    repo_ref = var.repo_ref
  })

  tags = merge(local.tags, { Name = "${var.name}-gpu-${count.index}" })
}

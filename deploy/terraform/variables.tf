variable "region" {
  description = "AWS region with GPU capacity for the chosen instance type."
  type        = string
  default     = "us-east-1"
}

variable "instance_type" {
  description = "GPU instance type. g5.12xlarge = 4x A10G (no NVLink, no EFA). Use g5.2xlarge (1 GPU) for the cheapest 2-node test, or p4d/p5 for NVLink + EFA."
  type        = string
  default     = "g5.12xlarge"
}

variable "instance_count" {
  description = "Number of GPU nodes."
  type        = number
  default     = 2
}

variable "key_name" {
  description = "Existing EC2 key pair name for SSH."
  type        = string
}

variable "ssh_cidr" {
  description = "CIDR allowed to SSH (your IP, e.g. 203.0.113.7/32). Never 0.0.0.0/0."
  type        = string
  validation {
    condition     = var.ssh_cidr != "0.0.0.0/0"
    error_message = "Refusing to open SSH to the internet; set ssh_cidr to your IP/32."
  }
}

variable "ami_id" {
  description = "Override AMI. Empty = latest AWS Deep Learning OSS Nvidia Driver AMI GPU PyTorch (Ubuntu 22.04)."
  type        = string
  default     = ""
}

variable "root_volume_gb" {
  type    = number
  default = 200
}

variable "repo_url" {
  description = "Git URL cloned onto each node by cloud-init."
  type        = string
  default     = "https://github.com/wenxuewu0216/gpu_watch.git"
}

variable "repo_ref" {
  type    = string
  default = "main"
}

variable "name" {
  type    = string
  default = "gpuwatch"
}

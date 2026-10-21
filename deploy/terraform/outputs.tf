output "public_ips" {
  value = aws_instance.gpu[*].public_ip
}

output "private_ips" {
  description = "Use these for NCCL / torchrun / Prometheus scrape targets."
  value       = aws_instance.gpu[*].private_ip
}

output "ssh" {
  value = [for ip in aws_instance.gpu[*].public_ip : "ssh ubuntu@${ip}"]
}

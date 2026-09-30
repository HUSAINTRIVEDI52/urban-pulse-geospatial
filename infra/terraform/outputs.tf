# ==============================================================================
# Terraform Outputs
# ==============================================================================

output "instance_id" {
  description = "OCID of the provisioned compute instance."
  value       = oci_core_instance.urbanpulse_vm.id
}

output "instance_public_ip" {
  description = "Public IPv4 address of the UrbanPulse server."
  value       = oci_core_instance.urbanpulse_vm.public_ip
}

output "instance_private_ip" {
  description = "Private IPv4 address of the UrbanPulse server within the VCN."
  value       = oci_core_instance.urbanpulse_vm.private_ip
}

output "ssh_connection_command" {
  description = "SSH terminal command to connect to the instance."
  value       = "ssh ubuntu@${oci_core_instance.urbanpulse_vm.public_ip}"
}

output "web_url" {
  description = "Public URL for the UrbanPulse web application."
  value       = "http://${oci_core_instance.urbanpulse_vm.public_ip}"
}

output "api_docs_url" {
  description = "Public URL for the FastAPI interactive Swagger documentation."
  value       = "http://${oci_core_instance.urbanpulse_vm.public_ip}/docs"
}

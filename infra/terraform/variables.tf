# ==============================================================================
# Oracle Cloud Infrastructure (OCI) Authentication & Tenancy Variables
# ==============================================================================

variable "tenancy_ocid" {
  description = "The OCID of the OCI tenancy."
  type        = string
}

variable "user_ocid" {
  description = "The OCID of the user calling the API."
  type        = string
}

variable "fingerprint" {
  description = "The fingerprint of the public API key uploaded to OCI."
  type        = string
}

variable "private_key_path" {
  description = "Local path to the OCI API private key file."
  type        = string
}

variable "compartment_ocid" {
  description = "The OCID of the compartment where resources will be provisioned."
  type        = string
}

variable "region" {
  description = "OCI region identifier (e.g., us-ashburn-1, ap-mumbai-1, eu-frankfurt-1)."
  type        = string
  default     = "ap-mumbai-1"
}

# ==============================================================================
# Network & Firewall Variables
# ==============================================================================

variable "vcn_cidr" {
  description = "CIDR block for the Virtual Cloud Network (VCN)."
  type        = string
  default     = "10.0.0.0/16"
}

variable "subnet_cidr" {
  description = "CIDR block for the public subnet."
  type        = string
  default     = "10.0.1.0/24"
}

variable "admin_cidr" {
  description = "Restricted CIDR block allowed for SSH (22) and Kubernetes API (6443) access."
  type        = string
  default     = "0.0.0.0/0"
}

# ==============================================================================
# Compute Instance Variables (Always Free Eligible)
# ==============================================================================

variable "instance_shape" {
  description = "OCI Compute shape. VM.Standard.A1.Flex is eligible for OCI Always Free tier (up to 4 OCPU, 24GB RAM)."
  type        = string
  default     = "VM.Standard.A1.Flex"
}

variable "instance_ocpus" {
  description = "Number of OCPUs for the Ampere A1 Flex instance."
  type        = number
  default     = 4
}

variable "instance_memory_in_gbs" {
  description = "RAM memory allocation in GBs for the Ampere A1 Flex instance."
  type        = number
  default     = 24
}

variable "boot_volume_size_in_gbs" {
  description = "Boot volume size in GB (Always Free allows up to 200GB total across tenancy)."
  type        = number
  default     = 100
}

variable "ssh_public_key" {
  description = "Public SSH key string to install on the instance for the default ubuntu/opc user."
  type        = string
}

variable "instance_display_name" {
  description = "Display name for the compute instance."
  type        = string
  default     = "urbanpulse-server"
}

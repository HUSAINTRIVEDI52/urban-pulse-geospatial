# ==============================================================================
# Data Sources: Availability Domains & Ubuntu OS Image
# ==============================================================================

data "oci_identity_availability_domains" "ads" {
  compartment_id = var.compartment_ocid
}

data "oci_core_images" "ubuntu_images" {
  compartment_id           = var.compartment_ocid
  operating_system         = "Canonical Ubuntu"
  operating_system_version = "22.04"
  shape                    = var.instance_shape
  sort_by                  = "TIMECREATED"
  sort_order               = "DESC"
}

# ==============================================================================
# Virtual Cloud Network (VCN) & Internet Gateway
# ==============================================================================

resource "oci_core_vcn" "urbanpulse_vcn" {
  compartment_id = var.compartment_ocid
  cidr_blocks    = [var.vcn_cidr]
  display_name   = "urbanpulse-vcn"
  dns_label      = "urbanpulse"

  freeform_tags = {
    "Project" = "UrbanPulse"
  }
}

resource "oci_core_internet_gateway" "urbanpulse_igw" {
  compartment_id = var.compartment_ocid
  vcn_id         = oci_core_vcn.urbanpulse_vcn.id
  display_name   = "urbanpulse-igw"
  enabled        = true

  freeform_tags = {
    "Project" = "UrbanPulse"
  }
}

resource "oci_core_route_table" "urbanpulse_rt" {
  compartment_id = var.compartment_ocid
  vcn_id         = oci_core_vcn.urbanpulse_vcn.id
  display_name   = "urbanpulse-public-rt"

  route_rules {
    destination       = "0.0.0.0/0"
    destination_type  = "CIDR_BLOCK"
    network_entity_id = oci_core_internet_gateway.urbanpulse_igw.id
  }

  freeform_tags = {
    "Project" = "UrbanPulse"
  }
}

# ==============================================================================
# Security List / Firewall (Ports 22, 80, 443, 6443)
# ==============================================================================

resource "oci_core_security_list" "urbanpulse_sl" {
  compartment_id = var.compartment_ocid
  vcn_id         = oci_core_vcn.urbanpulse_vcn.id
  display_name   = "urbanpulse-security-list"

  # Outbound egress - allow all traffic
  egress_security_rules {
    destination = "0.0.0.0/0"
    protocol    = "all"
    description = "Allow all outbound internet traffic"
  }

  # Ingress - SSH (Port 22, restricted to admin CIDR)
  ingress_security_rules {
    protocol    = "6" # TCP
    source      = var.admin_cidr
    description = "SSH Remote Administration"

    tcp_options {
      min = 22
      max = 22
    }
  }

  # Ingress - HTTP (Port 80)
  ingress_security_rules {
    protocol    = "6" # TCP
    source      = "0.0.0.0/0"
    description = "Public HTTP Web Traffic"

    tcp_options {
      min = 80
      max = 80
    }
  }

  # Ingress - HTTPS (Port 443)
  ingress_security_rules {
    protocol    = "6" # TCP
    source      = "0.0.0.0/0"
    description = "Public HTTPS Secure Web Traffic"

    tcp_options {
      min = 443
      max = 443
    }
  }

  # Ingress - Kubernetes API (Port 6443, restricted to admin CIDR)
  ingress_security_rules {
    protocol    = "6" # TCP
    source      = var.admin_cidr
    description = "k3s Kubernetes API Server"

    tcp_options {
      min = 6443
      max = 6443
    }
  }

  freeform_tags = {
    "Project" = "UrbanPulse"
  }
}

# ==============================================================================
# Public Subnet
# ==============================================================================

resource "oci_core_subnet" "urbanpulse_subnet" {
  compartment_id             = var.compartment_ocid
  vcn_id                     = oci_core_vcn.urbanpulse_vcn.id
  cidr_block                 = var.subnet_cidr
  display_name               = "urbanpulse-public-subnet"
  dns_label                  = "public"
  route_table_id             = oci_core_route_table.urbanpulse_rt.id
  security_list_ids          = [oci_core_security_list.urbanpulse_sl.id]
  prohibit_public_ip_on_vnic = false

  freeform_tags = {
    "Project" = "UrbanPulse"
  }
}

# ==============================================================================
# Always Free Compute Instance (Ampere A1 Arm64 Flex)
# ==============================================================================

resource "oci_core_instance" "urbanpulse_vm" {
  compartment_id      = var.compartment_ocid
  availability_domain = data.oci_identity_availability_domains.ads.availability_domains[0].name
  display_name        = var.instance_display_name
  shape               = var.instance_shape

  shape_config {
    ocpus         = var.instance_ocpus
    memory_in_gbs = var.instance_memory_in_gbs
  }

  source_details {
    source_type             = "image"
    source_id               = data.oci_core_images.ubuntu_images.images[0].id
    boot_volume_size_in_gbs = var.boot_volume_size_in_gbs
  }

  create_vnic_details {
    subnet_id        = oci_core_subnet.urbanpulse_subnet.id
    display_name     = "urbanpulse-primary-vnic"
    assign_public_ip = true
  }

  metadata = {
    ssh_authorized_keys = var.ssh_public_key
  }

  freeform_tags = {
    "Project" = "UrbanPulse"
  }
}

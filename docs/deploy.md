# UrbanPulse Cloud Deployment Guide

This guide describes how to deploy the complete **UrbanPulse** stack (PostGIS, FastAPI backend, Nginx frontend, automated weekly satellite pipeline, Traefik Ingress) to an **Oracle Cloud Infrastructure (OCI) Always Free** server from scratch using Terraform and Ansible.

---

## 1. Prerequisites

1. **Oracle Cloud Account**: Free tier eligible for Ampere A1 ARM64 shapes (up to 4 OCPUs, 24GB RAM).
2. **Local Tooling**:
   - [Terraform](https://developer.hashicorp.com/terraform/install) (>= 1.5.0)
   - [Ansible](https://docs.ansible.com/ansible/latest/installation_guide/intro_installation.html) (>= 2.15)
   - [OpenSSH Client](https://www.openssh.com/)
3. **OCI API Key**:
   - Generate an API key in your OCI Console (*Profile -> User Settings -> API Keys -> Add API Key*).
   - Note your **Tenancy OCID**, **User OCID**, **Compartment OCID**, and **Fingerprint**.

---

## 2. Step 1: Provision Cloud Infrastructure (Terraform)

```bash
# 1. Navigate to the terraform directory
cd infra/terraform

# 2. Copy the example configuration file
cp terraform.tfvars.example terraform.tfvars

# 3. Edit terraform.tfvars with your OCI OCIDs, SSH public key, and region
nano terraform.tfvars

# 4. Initialize Terraform and download the OCI provider plugin
terraform init

# 5. Plan and review the infrastructure changes
terraform plan

# 6. Provision the VCN, security list, subnet, and Ampere A1 compute instance
terraform apply
```

Upon completion, Terraform will output your server's public IP:
```
Outputs:
instance_public_ip      = "203.0.113.50"
ssh_connection_command  = "ssh ubuntu@203.0.113.50"
web_url                 = "http://203.0.113.50"
```

---

## 3. Step 2: Configure & Run Server Automation (Ansible)

```bash
# 1. Navigate to the ansible directory
cd ../ansible

# 2. Create your inventory file from the example
cp inventory.example.ini inventory.ini

# 3. Update the server IP address in inventory.ini:
#    [urbanpulse_servers]
#    server1 ansible_host=203.0.113.50 ansible_user=ubuntu ansible_ssh_private_key_file=~/.ssh/id_ed25519
nano inventory.ini

# 4. Execute the Ansible playbook:
#    - Hardens OS (creates non-root user, disables root password auth, enables UFW + fail2ban)
#    - Installs Docker CE and k3s Lightweight Kubernetes
#    - Deploys UrbanPulse manifests via Kustomize
ansible-playbook -i inventory.ini site.yml
```

---

## 4. Step 3: Verification & Access

Once Ansible completes:

1. **Web Dashboard**: Open `http://<SERVER_IP>/` in your browser.
2. **API Interactive Docs**: Open `http://<SERVER_IP>/docs`.
3. **Health Check**: Run `curl http://<SERVER_IP>/health`.
4. **Inspect Kubernetes Workloads**:
   ```bash
   ssh -i ~/.ssh/id_ed25519 urbanpulse@<SERVER_IP>
   kubectl get all -n urbanpulse
   ```

---

## 5. Day-2 Maintenance & Pipeline Triggering

### Run Satellite Pipeline On-Demand for a City:
```bash
# On the remote server:
kubectl create job --from=cronjob/urbanpulse-pipeline-weekly urbanpulse-run-manual -n urbanpulse
kubectl logs -f job/urbanpulse-run-manual -n urbanpulse
```

### Teardown Infrastructure:
```bash
cd infra/terraform
terraform destroy
```

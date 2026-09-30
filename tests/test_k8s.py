"""
Unit tests for UrbanPulse Kubernetes manifests and Kustomize overlays.
Validates YAML structure, required resources, probes, volume claims, and labels.
"""

from pathlib import Path

import yaml

K8S_BASE_DIR = Path(__file__).resolve().parent.parent / "infra" / "k8s" / "base"
K8S_LOCAL_DIR = Path(__file__).resolve().parent.parent / "infra" / "k8s" / "overlays" / "local"


def test_k8s_base_files_exist():
    """Validates that all required base Kubernetes manifest files exist."""
    expected_files = [
        "namespace.yaml",
        "secret.example.yaml",
        "configmap-cities.yaml",
        "configmap-schema.yaml",
        "pvc-data.yaml",
        "postgis.yaml",
        "api.yaml",
        "web.yaml",
        "ingress.yaml",
        "pipeline-cronjob.yaml",
        "kustomization.yaml",
    ]
    for filename in expected_files:
        filepath = K8S_BASE_DIR / filename
        assert filepath.exists(), f"Missing required manifest: {filename}"


def test_k8s_manifests_valid_yaml():
    """Parses every YAML manifest in base/ to ensure strict syntax correctness."""
    for yaml_path in K8S_BASE_DIR.glob("*.yaml"):
        with open(yaml_path, encoding="utf-8") as f:
            docs = list(yaml.safe_load_all(f))
            assert len(docs) > 0, f"Empty YAML file: {yaml_path.name}"
            for doc in docs:
                assert isinstance(doc, dict), f"Invalid document in {yaml_path.name}"
                assert "kind" in doc or "resources" in doc


def test_postgis_statefulset_structure():
    """Checks PostGIS StatefulSet specifications (probes, volume claims, port)."""
    postgis_path = K8S_BASE_DIR / "postgis.yaml"
    with open(postgis_path, encoding="utf-8") as f:
        docs = list(yaml.safe_load_all(f))

    statefulset = next(d for d in docs if d.get("kind") == "StatefulSet")
    spec = statefulset["spec"]
    assert spec["serviceName"] == "db"
    assert spec["replicas"] == 1

    container = spec["template"]["spec"]["containers"][0]
    assert container["image"].startswith("postgis/postgis")
    assert "livenessProbe" in container
    assert "readinessProbe" in container
    assert len(spec["volumeClaimTemplates"]) >= 1
    assert spec["volumeClaimTemplates"][0]["metadata"]["name"] == "pgdata"


def test_api_deployment_structure():
    """Validates API deployment has 2 replicas, probes on /health, and resource limits."""
    api_path = K8S_BASE_DIR / "api.yaml"
    with open(api_path, encoding="utf-8") as f:
        docs = list(yaml.safe_load_all(f))

    deployment = next(d for d in docs if d.get("kind") == "Deployment")
    spec = deployment["spec"]
    assert spec["replicas"] == 2

    container = spec["template"]["spec"]["containers"][0]
    assert "resources" in container
    assert "requests" in container["resources"]
    assert "limits" in container["resources"]
    assert container["livenessProbe"]["httpGet"]["path"] == "/health"
    assert container["readinessProbe"]["httpGet"]["path"] == "/health"


def test_web_and_ingress_structure():
    """Validates Web deployment and Traefik Ingress routing."""
    web_path = K8S_BASE_DIR / "web.yaml"
    with open(web_path, encoding="utf-8") as f:
        docs = list(yaml.safe_load_all(f))
    deployment = next(d for d in docs if d.get("kind") == "Deployment")
    assert (
        deployment["spec"]["template"]["spec"]["containers"][0]["ports"][0]["containerPort"] == 80
    )

    ingress_path = K8S_BASE_DIR / "ingress.yaml"
    with open(ingress_path, encoding="utf-8") as f:
        ingress = yaml.safe_load(f)

    assert ingress["kind"] == "Ingress"
    paths = [p["path"] for p in ingress["spec"]["rules"][0]["http"]["paths"]]
    assert "/" in paths
    assert "/api" in paths or "/health" in paths


def test_pipeline_cronjob_structure():
    """Validates weekly CronJob with Forbid concurrency and backoff limit."""
    cron_path = K8S_BASE_DIR / "pipeline-cronjob.yaml"
    with open(cron_path, encoding="utf-8") as f:
        cronjob = yaml.safe_load(f)

    assert cronjob["kind"] == "CronJob"
    assert cronjob["spec"]["concurrencyPolicy"] == "Forbid"
    assert cronjob["spec"]["jobTemplate"]["spec"]["backoffLimit"] == 2
    assert "successfulJobsHistoryLimit" in cronjob["spec"]


def test_kustomize_overlays():
    """Ensures local Kustomize overlay references base and provides secret generator."""
    local_kust_path = K8S_LOCAL_DIR / "kustomization.yaml"
    assert local_kust_path.exists()
    with open(local_kust_path, encoding="utf-8") as f:
        kust = yaml.safe_load(f)

    assert "../../base" in kust["resources"]
    assert "secretGenerator" in kust


def test_terraform_files_exist():
    """Validates that all required Terraform configuration files exist."""
    tf_dir = Path(__file__).resolve().parent.parent / "infra" / "terraform"
    expected = [
        "main.tf",
        "variables.tf",
        "outputs.tf",
        "providers.tf",
        "terraform.tfvars.example",
    ]
    for filename in expected:
        assert (tf_dir / filename).exists(), f"Missing Terraform file: {filename}"


def test_ansible_files_exist():
    """Validates that all required Ansible playbooks, roles, and inventory exist."""
    ansible_dir = Path(__file__).resolve().parent.parent / "infra" / "ansible"
    assert (ansible_dir / "site.yml").exists()
    assert (ansible_dir / "ansible.cfg").exists()
    assert (ansible_dir / "inventory.example.ini").exists()

    expected_roles = ["common", "security", "docker", "k3s", "deploy"]
    for role in expected_roles:
        role_task = ansible_dir / "roles" / role / "tasks" / "main.yml"
        assert role_task.exists(), f"Missing tasks for role: {role}"


def test_deploy_documentation_exists():
    """Validates that the cloud deployment documentation docs/deploy.md exists."""
    deploy_doc = Path(__file__).resolve().parent.parent / "docs" / "deploy.md"
    assert deploy_doc.exists()
    content = deploy_doc.read_text(encoding="utf-8")
    assert "terraform" in content.lower()
    assert "ansible" in content.lower()

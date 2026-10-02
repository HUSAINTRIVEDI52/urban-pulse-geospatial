# UrbanPulse Automation Makefile

.PHONY: help build up down logs test pipeline clean k3d-up k3d-deploy k3d-down

YEAR ?= 2024
CITY ?= ahmedabad
CLUSTER_NAME ?= urbanpulse

help:
	@echo "UrbanPulse Container & Kubernetes Commands:"
	@echo "  make build          - Build all Docker images"
	@echo "  make up             - Start web and API services in background (Docker Compose)"
	@echo "  make down           - Stop and remove all containers (Docker Compose)"
	@echo "  make logs           - View live container logs"
	@echo "  make test           - Run Pytest test suite"
	@echo "  make pipeline [YEAR=2024] [CITY=ahmedabad] - Run pipeline container for a given year"
	@echo "  make k3d-up         - Spin up local k3d Kubernetes cluster"
	@echo "  make k3d-deploy     - Build, import images, and apply Kustomize manifests to k3d"
	@echo "  make k3d-down       - Delete local k3d Kubernetes cluster"

build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f

test:
	pytest -v

pipeline:
	docker compose run --rm pipeline --city $(CITY) --year $(YEAR)

load-db:
	docker compose run --rm --entrypoint python pipeline -m pipeline.load_db --city $(CITY)

clean:
	docker compose down -v --remove-orphans

# ----------------------------------------------------------------------------
# Local Kubernetes (k3d) Workflows
# ----------------------------------------------------------------------------

k3d-up:
	k3d cluster create $(CLUSTER_NAME) \
		--port "8080:80@loadbalancer" \
		--port "8000:8000@loadbalancer" \
		--agents 1

k3d-deploy:
	docker build -t urbanpulse-api:latest -f docker/Dockerfile.api .
	docker build -t urbanpulse-web:latest -f docker/Dockerfile.web .
	docker build -t urbanpulse-pipeline:latest -f docker/Dockerfile.pipeline .
	k3d image import urbanpulse-api:latest urbanpulse-web:latest urbanpulse-pipeline:latest -c $(CLUSTER_NAME)
	kubectl apply -k infra/k8s/overlays/local

k3d-down:
	k3d cluster delete $(CLUSTER_NAME)

# ----------------------------------------------------------------------------
# Documentation & Reporting Targets
# ----------------------------------------------------------------------------

report:
	python pipeline/generate_report.py

report-pdf: report
	python pipeline/export_report_pdf.py

docs: report report-pdf
	@echo "[+] Documentation, HTML and PDF reports generated successfully."


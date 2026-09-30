# UrbanPulse Automation Makefile

.PHONY: help build up down logs test pipeline clean

YEAR ?= 2024
CITY ?= ahmedabad

help:
	@echo "UrbanPulse Container Management Commands:"
	@echo "  make build          - Build all Docker images"
	@echo "  make up             - Start web and API services in background"
	@echo "  make down           - Stop and remove all containers"
	@echo "  make logs           - View live container logs"
	@echo "  make test           - Run Pytest test suite"
	@echo "  make pipeline [YEAR=2024] [CITY=ahmedabad] - Run pipeline container for a given year"

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

clean:
	docker compose down -v --remove-orphans

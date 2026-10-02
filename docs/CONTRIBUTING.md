# Contributing to UrbanPulse

Thank you for your interest in contributing to **UrbanPulse**! We welcome contributions to satellite algorithms, API performance, UI visualizations, and infrastructure automation.

---

## 1. Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/husaintrivedi/UrbanPulse.git
   cd UrbanPulse
   ```

2. **Create Python virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Install pre-commit hooks**:
   ```bash
   pip install pre-commit ruff black
   pre-commit install
   ```

---

## 2. Code Quality & Standards

- **Formatting**: Format code using `black .`
- **Linting**: Check code with `ruff check .`
- **Unit Tests**: Ensure all tests pass using `pytest -v`
- **Infrastructure**: Validate Terraform with `terraform fmt -check` and Ansible with `ansible-lint`

---

## 3. How to Add a New City

UrbanPulse is completely configuration-driven. To onboard a new metropolitan area with zero city-specific code:

1. **Create Config File**:
   Create `configs/<city_name>.yaml` with your target coordinates and dry-season window:
   ```yaml
   city:
     name: Jaipur
     state: Rajasthan
     country: India
   spatial:
     bbox: [75.6000, 26.7000, 76.0500, 27.1500] # ~45x45 km AOI
     crs: EPSG:4326
     center_lat: 26.9124
     center_lon: 75.7873
     buffered_distance_km: 8.0
     aoi_size_km: 45.0
   temporal:
     analysis_years:
       start_year: 2018
       end_year: 2024
     dry_season:
       start_month: 11
       end_month: 2
       start_day: 11-01
       end_day: 02-28
   ```

2. **Run the Automated Pipeline**:
   ```bash
   python -m pipeline.run_city --city jaipur
   ```

3. **Verify Quality Gate & Documentation**:
   ```bash
   python pipeline/generate_report.py --city jaipur
   pytest -v
   ```

---

## 4. Submitting Pull Requests

1. Fork the repo and create a feature branch (`git checkout -b feature/amazing-feature`).
2. Verify all tests pass (`pytest -v`).
3. Open a Pull Request against `main`.

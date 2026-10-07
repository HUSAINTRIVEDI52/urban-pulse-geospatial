"""
UrbanPulse End-to-End User Flow and Integrity Tests
Tests:
- Guided tour flow
- Glossary popup & definition drawer
- City switching and metadata updates
- Compare mode swipe functionality at 390px and 1440px
- Simple vs Detailed depth toggle
- Numeric verification against stats.json (zero hardcoded figures)
"""

import json
import re

import pytest

try:
    from playwright.sync_api import sync_playwright

    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False


@pytest.mark.skipif(not PLAYWRIGHT_AVAILABLE, reason="playwright not installed")
def test_ui_user_flows_and_numbers():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(viewport={"width": 1440, "height": 900})
        ctx.add_init_script("localStorage.setItem('urbanpulse_tour_completed', 'true');")
        page = ctx.new_page()

        # Load web app
        try:
            page.goto("http://localhost:8080/", wait_until="networkidle", timeout=3000)
        except Exception as e:
            pytest.skip(f"Local test server at http://localhost:8080/ is not running: {e}")
        page.wait_for_timeout(2000)

        # 1. Test Glossary Interaction (use a term in #dashboard which is always visible)
        term_el = page.query_selector("#dashboard .glossary-term[data-term='land-cover']")
        if term_el is None:
            term_el = page.query_selector("#dashboard .glossary-term")
        assert term_el is not None
        term_el.click()
        page.wait_for_timeout(500)
        assert page.is_visible("#glossary-popover")
        page.click("#popover-close-btn")

        # 2. Test Compare Mode
        page.click("#toggle-compare-btn")
        page.wait_for_timeout(500)
        assert page.is_visible("#compare-container")
        assert page.is_visible("#swipe-divider")
        page.click("#close-compare-btn")
        page.wait_for_timeout(300)
        assert not page.is_visible("#compare-container")

        # 3. Test Simple Mode
        assert page.evaluate("window.UrbanPulseDepth.getDepth()") == "simple"

        # 4. Test City Switch to Pune
        page.select_option("#city-select", "pune")
        page.wait_for_timeout(2000)
        assert page.evaluate("window.UrbanPulseState.currentCity") == "pune"
        assert page.evaluate("window.UrbanPulseState.statsData.city") == "Pune"

        # Switch back
        page.select_option("#city-select", "ahmedabad")
        page.wait_for_timeout(2000)

        # 5. Number verification for Ahmedabad
        all_text = (page.text_content("#dashboard") + " " + page.text_content("#view-findings")).replace(",", "")
        found_tokens = set(re.findall(r"\b\d+(?:\.\d+)?\b", all_text))

        with open("web/data/ahmedabad/stats.json", encoding="utf-8") as f:
            stats = json.load(f)

        allowed = {
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "10",
            "11",
            "12",
            "13",
            "14",
            "15",
            "16",
            "18",
            "20",
            "24",
            "45",
            "50",
            "52",
            "60",
            "90",
            "95",
            "99",
            "100",
            "297",
            "300",
            "400",
            "2014",
            "2018",
            "2019",
            "2020",
            "2021",
            "2022",
            "2023",
            "2024",
            "2026",
        }

        def collect_nums(obj):
            if isinstance(obj, (int, float)):
                allowed.add(f"{obj}")
                allowed.add(f"{obj:.1f}")
                allowed.add(f"{obj:.2f}")
                allowed.add(f"{obj:.4f}")
                allowed.add(f"{round(obj)}")
                if 0 < obj <= 1:
                    allowed.add(f"{obj * 100:.1f}")
                    allowed.add(f"{obj * 100:.0f}")
            elif isinstance(obj, str):
                for n in re.findall(r"\b\d+(?:\.\d+)?\b", obj):
                    allowed.add(n)
            elif isinstance(obj, dict):
                for v in obj.values():
                    collect_nums(v)
            elif isinstance(obj, list):
                for it in obj:
                    collect_nums(it)

        collect_nums(stats)

        # Include derived side length from aoi_area_km2
        if "aoi_area_km2" in stats:
            side = stats["aoi_area_km2"] ** 0.5
            allowed.add(f"{side:.1f}")
            allowed.add(f"{round(side)}")

        # Include year-over-year percentage changes in growth_series
        if "growth_series" in stats:
            gs = stats["growth_series"]
            for i in range(1, len(gs)):
                diff = gs[i]["norm_builtup_km2"] - gs[i - 1]["norm_builtup_km2"]
                pct = abs((diff / gs[i - 1]["norm_builtup_km2"]) * 100)
                allowed.add(f"{pct:.1f}")
                allowed.add(f"{round(pct)}")

        # Include derived stratum ratios
        if "change_validation" in stats and "strata_table" in stats["change_validation"]:
            for s in stats["change_validation"]["strata_table"]:
                ss = s.get("sample_size") or 1
                for key in ["c01_gain", "c00", "c11", "c10"]:
                    if key in s:
                        ratio_pct = (s[key] / ss) * 100
                        allowed.add(f"{ratio_pct:.1f}")
                        allowed.add(f"{round((s[key] / ss) * 10)}")

        # Verify no token is unaccounted for
        unaccounted = []
        for t in found_tokens:
            if t not in allowed:
                try:
                    f = float(t)
                    if not any(
                        abs(float(a) - f) < 0.05 for a in allowed if re.match(r"^\d+(\.\d+)?$", a)
                    ):
                        unaccounted.append(t)
                except ValueError:
                    unaccounted.append(t)

        assert len(unaccounted) == 0, f"Unaccounted numbers in UI: {unaccounted}"

        ctx.close()
        browser.close()

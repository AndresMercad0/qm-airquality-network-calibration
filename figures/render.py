"""
HTML to PNG capture with headless Chrome

Finds the Chrome binary (AQMS_CHROME, the macOS application or a Linux
binary on the PATH) and captures the figure-root element of an HTML file
as a PNG through Selenium. Imported by the figure scripts.
"""

import base64
import os
import shutil
import time
from pathlib import Path

# -------------------------------------------------------------------------
# Chrome binary
# -------------------------------------------------------------------------
MACOS_CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
LINUX_CHROME_NAMES = ("google-chrome", "chromium", "chromium-browser")


def find_chrome():
    """Return the Chrome binary to drive, or None to let Selenium locate one."""
    from_env = os.environ.get("AQMS_CHROME")
    if from_env:
        return from_env
    if Path(MACOS_CHROME).exists():
        return MACOS_CHROME
    for name in LINUX_CHROME_NAMES:
        found = shutil.which(name)
        if found:
            return found
    return None


# -------------------------------------------------------------------------
# HTML to PNG
# -------------------------------------------------------------------------
def render_with_selenium(html_path: Path, png_path: Path, cfg, wait_s=2.4):
    """Capture the #figure-root element of an HTML file as a PNG with headless Chrome."""
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options

    vp = cfg["viewport"]
    w = vp["width_px"]
    h = vp["height_px"]
    dsf = vp.get("device_scale_factor", 2)

    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--hide-scrollbars")
    options.add_argument(f"--window-size={w + 40},{h + 40}")
    options.add_argument("--no-sandbox")
    chrome_path = find_chrome()
    if chrome_path:
        options.binary_location = chrome_path

    driver = webdriver.Chrome(options=options)
    try:
        driver.get(html_path.resolve().as_uri())
        time.sleep(wait_s)    # Lets the web fonts load before the capture
        box = driver.execute_script(
            "const r = document.getElementById('figure-root').getBoundingClientRect();"
            "return {x: r.x, y: r.y, w: r.width, h: r.height};"
        )
        clip = {
            "x": float(box["x"]),
            "y": float(box["y"]),
            "width": float(box["w"]),
            "height": float(box["h"]),
            "scale": dsf,
        }
        result = driver.execute_cdp_cmd("Page.captureScreenshot", {
            "format": "png",
            "clip": clip,
            "captureBeyondViewport": True,
            "fromSurface": True,
        })
        png_bytes = base64.b64decode(result["data"])
    finally:
        driver.quit()

    png_path.write_bytes(png_bytes)

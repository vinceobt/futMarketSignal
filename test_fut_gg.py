from patchright.sync_api import sync_playwright
import re

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto('https://www.fut.gg/players/239085-erling-haaland/')
    
    # Dump it to a file so we can inspect it easily
    with open('haaland.html', 'w') as f:
        f.write(page.content())
    browser.close()

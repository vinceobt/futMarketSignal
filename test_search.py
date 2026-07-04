from patchright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto('https://www.fut.gg/api/fut/players/search/?query=Erling%20Haaland')
    print(page.content()[:300])
    browser.close()

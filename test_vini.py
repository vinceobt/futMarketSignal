from patchright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto('https://www.fut.gg/players/?name=Vinicius')
    page.wait_for_timeout(2000)
    links = page.locator('a').all()
    for link in links:
        url = link.get_attribute('href')
        if url and '/players/' in url and url.count('/') >= 4:
            print(url)
    browser.close()

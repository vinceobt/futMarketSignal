from patchright.sync_api import sync_playwright

queries = [
    "Erling Haaland",
    "Kylian Mbappe",
    "Jude Bellingham",
    "Mohamed Salah",
    "Vinicius Jr",
    "Lamine Yamal",
    "Virgil van Dijk",
    "Rodri",
    "Jamie Vardy",
    "Sebastien Haller",
    "Memphis Depay",
    "Niklas Sule"
]

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()

    for q in queries:
        url_q = q.replace(' ', '+')
        page.goto(f'https://www.fut.gg/players/?name={url_q}')
        page.wait_for_timeout(2000)
        
        # Look for the player anchor tags that have the name inside them or check text
        links = page.locator('a').all()
        for link in links:
            url = link.get_attribute('href')
            if url and '/players/' in url and '-' in url and url.count('/') >= 4:
                print(f"{q}: -> {url}")
                # We want all versions because we want to grab base vs FOF
                # Let's print out the text to see ratings
        print("=========")

    browser.close()

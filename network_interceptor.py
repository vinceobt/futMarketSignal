from patchright.sync_api import sync_playwright
import time

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()

    prices = []
    
    def handle_response(response):
        if 'api' in response.url and 'price' in response.url:
            print("Found price URL:", response.url)
            try:
                print(response.json())
            except:
                pass

    page.on("response", handle_response)
    page.goto('https://www.fut.gg/players/239085-erling-haaland/')
    
    # wait a bit for async stuff
    time.sleep(3)
    browser.close()

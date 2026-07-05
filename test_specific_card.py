from patchright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()

    def handle_response(response):
        if 'api' in response.url and 'price' in response.url:
            print("Found price URL:", response.url)
            try:
                print(response.json())
            except:
                pass

    page.on("response", handle_response)
    page.goto('https://www.fut.gg/players/239085-erling-haaland/26-184788461/')
    page.wait_for_timeout(3000)
    
    browser.close()

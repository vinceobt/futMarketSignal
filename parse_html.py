from bs4 import BeautifulSoup

with open('haaland.html') as f:
    soup = BeautifulSoup(f, 'html.parser')

print("Title:", soup.title.string)
# Try to find elements with class containing 'price'
for el in soup.find_all(class_=lambda c: c and 'price' in c.lower()):
    print("Class:", el.get('class'), "Text:", el.get_text(strip=True))


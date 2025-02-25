import requests
from bs4 import BeautifulSoup, Tag
from bs4.element import NavigableString
import time

"""
     Fetches and extracts the main heading (H1) from the <article> section of a given webpage.

     Args:
         url (str): The URL of the webpage to scrape.

     Returns:
         str: The extracted H1 content if found, otherwise an error message.
"""
def scrape_url(url):
    start_time = time.perf_counter()
    try:
        resp = requests.get(url)
        resp.raise_for_status()
    except Exception as e:
        return f"Error fetching the URL: {e}"

    soup = BeautifulSoup(resp.content, 'html.parser')

    # Find the <article> section
    article_section = soup.find('article')
    if not isinstance(article_section, Tag):  # Ensure it's a valid tag
        return "No <article> section found on this page."

    # Extract relevant tags only within <article>
    allowed_tags = ['h1']
    extracted_content = []

    for tag in article_section.find_all(allowed_tags):
        if not isinstance(tag, Tag):  # Ensure tag is a proper BeautifulSoup Tag
            continue

        # Remove images inside the tags
        for img in tag.find_all('img'):
            img.decompose()

        # Convert <a> links to plain text
        for a in tag.find_all('a'):
            if isinstance(a, Tag) and 'href' in a.attrs:
                # Ensures replace_with receives valid input
                a.replace_with(NavigableString(a.get_text()))

        extracted_content.append(str(tag))

    content = " ".join(extracted_content)
    elapsed = time.perf_counter() - start_time
    print(f"[Scraping] Time taken: {elapsed * 1000:.2f} ms")

    return content  # Ensure function returns a string, not a list

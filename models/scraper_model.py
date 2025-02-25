import requests
from bs4 import BeautifulSoup, Tag
import time


def scrape_url(url):
    """
    Fetches and extracts the main heading (H1) from the <article> section of a given webpage.

    Args:
        url (str): The URL of the webpage to scrape.

    Returns:
        str: The extracted H1 text if found, otherwise an error message.
    """
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

    # Extract H1 text from <article>
    h1_tag = article_section.find('h1')
    if not h1_tag or not isinstance(h1_tag, Tag):
        return "No H1 found in the article."

    # Extract only the visible text from the <h1> tag
    # Strip removes extra spaces & line breaks
    h1_text = h1_tag.get_text(strip=True)

    # TODO: Remove this
    print("Scraped Title: ", h1_text)

    elapsed = time.perf_counter() - start_time
    print(f"[Scraping] Time taken: {elapsed * 1000:.2f} ms")

    return h1_text  # Returns only the raw text, no HTML

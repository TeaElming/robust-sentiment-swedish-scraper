from fastapi import APIRouter
from pydantic import BaseModel
from models.scraper_model import scrape_url
from models.sentiment_model import analyse_sentiment
from models.tokenizer_model import tokenize_text
import time

router = APIRouter()


class TextInput(BaseModel):
    url: str


@router.post("/scrape-sentiment")
def tokenize_and_analyse(data: TextInput):
    """
    Receives a URL, scrapes the H1 element, tokenizes it, and returns sentiment analysis.
    """
    total_start_time = time.perf_counter()

    # Step 1: Retrieve H1 text from the URL
    text_content = scrape_url(data.url)

    # Handle errors from scraper
    if text_content.startswith("Error") or text_content.startswith("No <article>"):
        return {"error": text_content}

    if not text_content.strip():  # Check if text is empty
        return {"error": "Extracted text is empty. Cannot perform sentiment analysis."}

    # Step 2: Tokenize the extracted H1 text
    tokenized = tokenize_text(text_content)

    # Ensure tokenization succeeded
    if "input_ids" not in tokenized or "attention_mask" not in tokenized:
        return {"error": "Tokenization failed. No valid input for sentiment analysis."}

    # Step 3: Perform Sentiment Analysis
    sentiment = analyse_sentiment(
        tokenized["input_ids"], tokenized["attention_mask"]
    )

    total_elapsed = time.perf_counter() - total_start_time
    print(f"[Total Request] Time taken: {total_elapsed * 1000:.2f} ms")

    return sentiment

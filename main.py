from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from routes.sentiment_route import router as sentiment_router

app = FastAPI()

# Allow CORS (Cross-Origin Requests)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from all sources (use specific origins in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers for modular API structure
app.include_router(sentiment_router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)

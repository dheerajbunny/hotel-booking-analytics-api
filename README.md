
Hotel Booking Analytics & Q&A API

Overview
This project provides an API for hotel booking analytics and a retrieval-augmented question-answering (QA) system. It uses machine learning to retrieve insights from booking data.

Features
Analytics Endpoint (/analytics): Provides revenue trends, cancellation rates, and top booking countries.
Q&A Endpoint (/ask): Uses FAISS & Sentence Transformers to answer booking-related queries.
Performance Evaluation: Measures API response time and accuracy.
Installation & Setup
Step 1: Install Dependencies
Run the following command to install all required dependencies:
pip install -r requirements.txt
Step 2: Run the API (Kaggle or Local Machine)
Start the API server by running:
python app.py
This will start the API on http://127.0.0.1:7500.

API Endpoints
1. POST /analytics
Returns analytics reports on hotel bookings.

Example Request:
import requests
response = requests.post("http://127.0.0.1:7500/analytics")
print(response.json())

Example Response:
{
  "revenue_trends": {"2017-06-01": 710153.16},
  "cancellation_rate": "37.04%",
  "top_booking_countries": {"PRT": 48590, "GBR": 12129},
  "response_time": "0.2564 sec"
}
2. POST /ask
Answers booking-related questions.

Example Request:
question = {"question": "What is the average price of a hotel booking?"}
response = requests.post("http://127.0.0.1:7500/ask", json=question)
print(response.json())

Example Response:
{
  "answer": {"adr": 100.0, "country": "PER"},
  "response_time": "0.1132 sec"
}
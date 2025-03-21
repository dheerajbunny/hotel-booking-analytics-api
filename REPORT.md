Implementation & Challenges

1. Implementation Choices

1.1 Data Cleaning
Handled missing values (children, country, agent, company).
Converted date columns for better analysis.
Created new features:
total_guests = adults + children + babies
total_nights = stays_in_weekend_nights + stays_in_week_nights

1.2 Analytics Calculations
Revenue Trends Over Time → Summed adr (average daily rate) per month.
Cancellation Rate → Computed as a percentage of canceled bookings.
Booking Distribution → Counted bookings per country.

1.3 Retrieval-Augmented Q&A (RAG)
Used sentence-transformers for text embeddings.
Implemented FAISS for fast similarity search on hotel booking queries.

1.4 Flask API
/analytics → Provides revenue & booking trends.
/ask → Answers booking-related queries using FAISS retrieval.

2. Challenges & Solutions

2.1 JSON Serialization Issues
Issue: Flask could not serialize int64, float64, and Timestamp data types.
Solution: Converted them to standard Python types (int, float, str).

2.2 Running Flask in Kaggle
Issue: Flask does not run interactively in Kaggle.
Solution: Used threading to keep Flask running inside Kaggle notebooks.

2.3 Performance Optimization
Logged API response times for tracking efficiency.
Used FAISS indexing for fast retrieval of relevant queries.
# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
import faiss
import requests
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import threading

# Load the dataset
file_path = "/kaggle/input/dsfsfsfs/hotel_bookings.csv"  # Update the path if needed
df = pd.read_csv(file_path)

# Step 1: Handle missing values
df['children'].fillna(0, inplace=True)
df['country'].fillna('Unknown', inplace=True)
df['agent'].fillna(0, inplace=True)
df['company'].fillna(0, inplace=True)

# Convert agent and company columns to integers
df['agent'] = df['agent'].astype(int)
df['company'] = df['company'].astype(int)

# Step 2: Convert date columns
df['reservation_status_date'] = pd.to_datetime(df['reservation_status_date'])

# Step 3: Create new features
df['total_guests'] = df['adults'] + df['children'] + df['babies']
df['total_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']

# Save the cleaned data
df.to_csv("hotel_bookings_cleaned.csv", index=False)
print("Data Cleaning Completed. Cleaned file saved as 'hotel_bookings_cleaned.csv'")

# Step 4: Revenue Trends Over Time
df['arrival_date'] = pd.to_datetime(df[['arrival_date_year', 'arrival_date_month']]
                                    .astype(str).agg('-'.join, axis=1))
monthly_revenue = df.groupby('arrival_date')['adr'].sum()

plt.figure(figsize=(12, 6))
sns.lineplot(x=monthly_revenue.index, y=monthly_revenue.values)
plt.xlabel("Month")
plt.ylabel("Total Revenue")
plt.title("Revenue Trends Over Time")
plt.xticks(rotation=45)
plt.show()

# Step 5: Compute Cancellation Rate
cancellation_rate = df['is_canceled'].mean() * 100
print(f"Cancellation Rate: {cancellation_rate:.2f}%")

# Step 6: Geographical Distribution of Bookings
country_counts = df['country'].value_counts()
plt.figure(figsize=(12, 6))
sns.barplot(x=country_counts.index[:10], y=country_counts.values[:10])
plt.xticks(rotation=45)
plt.xlabel("Country")
plt.ylabel("Number of Bookings")
plt.title("Top 10 Booking Countries")
plt.show()

# Step 7: Booking Lead Time Distribution
plt.figure(figsize=(12, 6))
sns.histplot(df['lead_time'], bins=50, kde=True)
plt.xlabel("Lead Time (Days)")
plt.ylabel("Number of Bookings")
plt.title("Booking Lead Time Distribution")
plt.show()

print("Analytics Calculations Completed!")

# Implement Retrieval-Augmented Q&A (RAG)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Define a few sample questions
questions = [
    "Show me total revenue for July 2017.",
    "Which locations had the highest booking cancellations?",
    "What is the average price of a hotel booking?"
]

# Convert questions to embeddings
question_embeddings = model.encode(questions)

# Convert dataset text columns into embeddings
dataset_text = df[['arrival_date_month', 'country', 'adr']].astype(str).agg(' '.join, axis=1)
dataset_embeddings = model.encode(dataset_text.tolist())

# Indexing with FAISS
index = faiss.IndexFlatL2(dataset_embeddings.shape[1])
index.add(np.array(dataset_embeddings))

# Function to find similar entries using FAISS
def query_rag(question):
    start_time = time.time()
    question_embedding = model.encode([question])
    D, I = index.search(np.array(question_embedding), 1)

    # Convert int64 and float64 to standard Python types for JSON serialization
    result = df.iloc[I[0][0]].to_dict()
    result = {k: (int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v)
              for k, v in result.items()}

    response_time = time.time() - start_time
    result["response_time"] = f"{response_time:.4f} sec"
    return result  # Convert result to dictionary

print("Retrieval-Augmented Q&A System Initialized!")

# Flask API Setup
app = Flask(__name__)

# API endpoint for question answering
@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.get_json()
    question = data.get("question", "")
    try:
        result = query_rag(question)
        return jsonify({"answer": dict(result), "response_time": result["response_time"]})
    except Exception as e:
        print(f"Error in /ask: {e}")
        return jsonify({"error": str(e)}), 500

# API endpoint for analytics
@app.route("/analytics", methods=["POST"])
def get_analytics():
    start_time = time.time()
    try:
        print("Generating analytics...")  # Debugging
        analytics_data = {
            "revenue_trends": {str(k): v for k, v in monthly_revenue.to_dict().items()},
            "cancellation_rate": f"{cancellation_rate:.2f}%",
            "top_booking_countries": country_counts[:10].to_dict()
        }
        response_time = time.time() - start_time
        analytics_data["response_time"] = f"{response_time:.4f} sec"
        print("Analytics data:", analytics_data)  # Debugging
        return jsonify(analytics_data)
    except Exception as e:
        print(f"Error in /analytics: {e}")  # Print error in logs
        return jsonify({"error": str(e)}), 500

# Run Flask API in a separate thread for Kaggle compatibility
def run_flask():
    app.run(host="0.0.0.0", port=7500, debug=False, use_reloader=False)

thread = threading.Thread(target=run_flask)
thread.start()

time.sleep(3)  # Give it a few seconds to start
print("Flask API Running! Test it locally in Kaggle using requests.")

# Test API endpoints locally
response = requests.post("http://127.0.0.1:7500/analytics")
print(response.json())

question = {"question": "What is the average price of a hotel booking?"}
response = requests.post("http://127.0.0.1:7500/ask", json=question)
print(response.json())

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import redis
import sys
import time
import warnings
from helpers import CacheEvaluator, PerfEval, SemanticCacheWrapper
from sentence_transformers import SentenceTransformer

# Suppress warnings
warnings.filterwarnings('ignore')

REDIS_URL = "redis://localhost:6379"


print("Redis connection check\n----------------------\n")

# Check if Redis is running
try:
    client = redis.from_url(REDIS_URL)
    client.ping()
    print("Redis connection successful")
    print(f"Connected to: {REDIS_URL}")
except Exception as e:
    print(f"Error: {e}")
    sys.exit("Redis connection failed! Please start Redis with: docker run -d --name redis -p 6379:6379 redis/redis-stack:latest")


print("\n\nSTEP 1 - Redis Setup & Vector Index\n-----------------------------------\n")

try:
    cache = SemanticCacheWrapper(
        name="ecommerce-support-cache",
        distance_threshold=0.13,
        ttl=604800, # 7 days in seconds
        redis_url=REDIS_URL
    )
    print("✅ Redis semantic cache initialised successfully")
    
    faq_df = pd.DataFrame({
        "question": [
            "How do I get a refund?",
            "Can I reset my password?",
            "Where is my order?",
            "How long is the warranty?",
            "Do you ship internationally?",
            "How do I cancel my subscription?",
            "Can I change my delivery address?",
            "What payment methods do you accept?"
        ],
        "answer": [
            "To request a refund, visit your orders page and select Request Refund. Refunds are processed within 3-5 business days.",
            "Click Forgot Password on the login page and follow the email instructions. Contact support if you don't receive the email within 10 minutes.",
            "Use the tracking link sent to your email after purchase. Allow 24-48 hours for tracking to activate.",
            "All electronic products include a 12-month warranty. Extended warranties are available at checkout.",
            "Yes, we ship to over 50 countries worldwide. International shipping typically takes 7-14 business days.",
            "Go to Account Settings > Subscriptions and click Cancel Subscription. You'll retain access until the end of your billing period.",
            "Yes, you can update your delivery address in Account Settings > Addresses before your order ships.",
            "We accept all major credit cards, PayPal, and Apple Pay. Some regions also support local payment methods."
        ]
    })
    
    # Load data into the cache (clears existing entries by default):
    cache.hydrate_from_df(faq_df)
    print("✅ FAQ data successfully hydrated into cache")
    
    # Check cache
    results = cache.check("I want my money back")
    if results.matches:
        print(f"Cache hit: {results.matches[0].response}")
    
except Exception as e:
    print("❌ Failed to initialise Redis semantic cache")
    print(f"Error: {e}")


print("\n\nSTEP 2 - Embedding Model Selection\n----------------------------------\n")

model = SentenceTransformer('all-MiniLM-L6-v2')

def encode_query(text: str):
    return model.encode(text)

sample_query = "How do I return an item?"
embedding = encode_query(sample_query)

print("Embedding shape:", embedding.shape)
print("First 5 values:", embedding[:5])
assert embedding.shape[0] == 384, "Embedding dimension does not match spec (384)"


print("\n\nSTEP 3 - Build Semantic Cache logic\n-----------------------------------\n")

hit_count = 0
miss_count = 0

def get_cached_or_generate(query: str, generate_fn):
    global hit_count, miss_count
    results = cache.check(query)
    if results.matches:
        hit_count += 1
        print(f"✅ Cache HIT for query: {query}")
        response = results.matches[0].response
    else:
        miss_count += 1
        print(f"❌ Cache MISS for query: {query}")
        response = generate_fn(query)
        cache.store(prompt=query, response=response)
    total = hit_count + miss_count
    hit_rate = hit_count / total if total > 0 else 0
    print(f"Hits: {hit_count}, Misses: {miss_count}, Hit Rate: {hit_rate:.2%}")
    return response

def mock_generate_fn(query: str):
    return f"Generated response for '{query}'"

sample_queries = [
    "How can I get a refund?",
    "I forgot my password",
    "What's your refund policy?",
    "Cancel my subscription",
    "Do you ship internationally?"
]

for q in sample_queries:
    response = get_cached_or_generate(q, mock_generate_fn)
    print(f"Response: {response}\n")


print("\n\nSTEP 4 - Cache Retrieval & Threshold Tuning\n-------------------------------------------\n")

test_queries = [
    "I want my money back",
    "Can I get a refund?",
    "What's your refund policy?",
    "Help me reset my password",
    "Where's my package?",
    "Stop my subscription",
    "Update my shipping info",
    "What payment do you take?",
    "What are your business hours?",
    "Do you have a mobile app?"
]

true_labels = [True, True, True, True, True, True, True, True, False, False]

def test_threshold(threshold: float, queries: list):
    results = cache.check_many(queries, distance_threshold=threshold)
    hits = sum(1 for r in results if r.matches)
    total = len(queries)
    hit_rate = hits / total if total > 0 else 0
    matched = [r.query for r in results if r.matches]
    print(f"Threshold: {threshold:.2f} | Hit Rate: {hit_rate:.2%} | Hits: {len(matched)} | Matched: {matched[:3]}")
    return hit_rate, results

thresholds = [0.08, 0.10, 0.13, 0.15, 0.18, 0.20, 0.25]
hit_rates = []
f1_scores = []

for t in thresholds:
    hr, results = test_threshold(t, test_queries)
    hit_rates.append(hr)
    evaluator = CacheEvaluator(true_labels, results)
    metrics = evaluator.get_metrics(distance_threshold=t)
    f1_scores.append(metrics["f1_score"])

plt.figure(figsize=(8,5))
plt.plot(thresholds, hit_rates, marker='o', label='Hit Rate')
plt.plot(thresholds, f1_scores, marker='s', label='F1 Score')
plt.title("Cache Hit Rate and F1 Score vs. Distance Threshold")
plt.xlabel("Distance Threshold")
plt.ylabel("Score")
plt.grid(True)
plt.legend()
plt.savefig("cache_hitrate_f1_distance_threshold.png")
plt.show(block=False)

best_idx = f1_scores.index(max(f1_scores))
print(f"Recommended optimal threshold: {thresholds[best_idx]:.2f} (F1 Score={f1_scores[best_idx]:.2%})")


print("\n\nSTEP 5 - Test & Evaluate Performance\n------------------------------------\n")

test_queries = [
    "How do I get a refund?",
    "What's your refund policy?",
    "Can I get a refund for my purchase?",
    "I want my money back",
    "How do I cancel my order?",
    "Where is my order?",
    "My package hasn’t arrived yet",
    "Track my shipment",
    "How can I reset my password?",
    "I forgot my password",
    "Help me log into my account",
    "Change my email address",
    "Update my shipping address",
    "Do you ship internationally?",
    "International shipping options",
    "When will my item ship?",
    "Estimated delivery time",
    "Can I change my billing information?",
    "Billing address update",
    "Do you offer gift wrapping?",
    "How long is the warranty?",
    "Product warranty duration",
    "Do you have a store near me?",
    "Store locations",
    "Can I get a student discount?",
    "Do you offer military discounts?",
    "Cancel subscription",
    "Stop recurring payments",
    "Resume my subscription",
    "How to change subscription plan",
    "Accept PayPal?",
    "Payment methods available",
    "Credit card not working",
    "Failed payment help",
    "Add a new payment method",
    "Refund taking too long",
    "Return shipping cost",
    "Return an item",
    "Where’s my confirmation email?",
    "Did my order go through?",
    "Order confirmation status",
    "Can I buy a gift card?",
    "How to use a promo code?",
    "My promo code isn’t working",
    "How to contact support?",
    "What are your customer service hours?",
    "Customer support phone number",
    "Do you have a mobile app?",
    "How do I delete my account?",
    "Request account deletion",
    "Privacy policy information",
    "Terms of service"
]

def mock_llm_response(prompt):
    time.sleep(np.random.uniform(0.4, 0.6))
    return f"Response generated for: {prompt}"

perf = PerfEval()
perf.set_total_queries(len(test_queries))

cache_hits = 0

with perf:
    for query in test_queries:
        perf.start()
        results = cache.check(query)
        if results.matches:
            perf.tick("cache_hit")
            cache_hits += 1
        else:
            perf.tick("cache_miss")
            perf.start()
            response = mock_llm_response(query)
            perf.tick("llm_call")
            perf.record_llm_call("gpt-4o-mini", query, response)
            cache.store(prompt=query, response=response)

summary = perf.summary(labels=["cache_hit", "llm_call"])
print(summary)

hit_rate = cache_hits / len(test_queries)
print(f"Cache Hit Rate: {hit_rate:.2%}")
if hit_rate >= 0.65:
    print("✅ Hit rate meets or exceeds 65% target.")
else:
    print("⚠️ Hit rate below target. Consider tuning similarity threshold.")

# Semantic Caching with Redis for cheaper and faster responses

Semantic Caching intelligently reuses responses to previous semantically similar queries. This is extremely important for Retrieval-Augmented Generation (RAG) systems, as well as agents and agentic systems.

Before each Large Language Model (LLM) call, Semantic Caching checks if the answer can be delivered from cached past responses. Cached responses are both cheaper and faster, as newer and bigger LLMs tend to become more expensive and often take longer to answer.

If someone asks "How can I get a refund?" and someone else says "I want my money back", then Semantic Caching can deliver the answer from the cache after it learns the first answer, because both sentences have the same meaning (semantic similarity), although they use different words.

![alt text](https://github.com/user-attachments/assets/8c799945-328c-42d5-9e9a-3ef43806c8ae "Semantic Caching")

Semantic Caching is a powerful technique for:

- Reducing operational costs (70%+ savings possible)
- Improving user experience (80%+ faster responses)
- Scaling Artificial Intelligence (AI) applications efficiently

The key is finding the right balance between hit rate (more cache hits) and precision (correct responses). Monitor both metrics continuously and adjust thresholds based on user feedback.

## What this example does

This example shows an intelligent Semantic Caching system using Redis.

Redis is a fast, open-source data store that keeps information in memory instead of on disk, making it extremely quick. Redis can also be used as a key/value vector store for semantic search.

This example simulates a Semantic Caching system for e-commerce support. The purpose of this system is to cache responses to customer support questions, reducing LLM Application Programming Interface (API) costs, and improving response times. The system handles refund requests, shipping inquiries, product questions, and account issues.

The Semantic Caching system is built in 5 steps:

1. **Redis Setup** 🔧 - Configure Redis vector store with embeddings
2. **Embedding Model** 🧠 - Select and implement semantic embeddings
3. **Build Cache** 💾 - Implement core Semantic Caching logic
4. **Cache Retrieval** 🔍 - Query similarity and threshold logic
5. **Test & Evaluate** 📊 - Measure performance and hit rates

Please note that when Redis uses the term "cache", it refers to the Redis in-memory key/value store with additional features, such as automatic expiration using Time-To-Live (TTL) in seconds.

This example is a demonstration that only simulates LLM calls. The LLM response times are simulated using a random delay between 0.4 and 0.6 seconds (400-600 ms). In a real-world implementation, this would have to be replaced by real LLM calls.

## Key quality evaluation concepts

### Distance vs Similarity

- **Vector Distance**: Lower = more similar (0.0 = identical, 2.0 = opposite for cosine)
- **Cosine Similarity**: Higher = more similar (1.0 = identical, 0.0 = orthogonal)

Conversion:

> similarity = (2 - distance) / 2

### Threshold selection

| Threshold | Similarity | Behaviour                    |
| --------- | ---------- | ---------------------------- |
| 0.05-0.10 | 95-97%     | Very strict, high precision  |
| 0.10-0.15 | 92-95%     | Balanced (recommended)       |
| 0.15-0.25 | 87-92%     | Loose, high hit rate         |
| 0.25-0.30 | 85-87%     | Default, good starting point |

### Precision vs Recall trade-off

- **Precision**: How many cache hits were correct?
- **Recall**: How many correct matches were found?
- **F1 Score**: Balanced metric (harmonic mean)

Lower threshold means higher precision, lower recall
Higher threshold means lower precision, higher recall

### Cache Hit Rate vs Utility

- **Hit Rate**: Fraction of queries that hit cache
- **Utility**: Harmonic mean of precision and hit rate

Optimise for utility to balance cost savings (hit rate) with accuracy (precision).

![alt text](https://github.com/user-attachments/assets/8b8e508a-c915-4d93-9e91-571d642646d7 "Threshold Tuning")

## Embedding model

The embedding model used in this example is `all-MiniLM-L6-v2`. This is a fast, general purpose embedding model with 384 dimensions (dims).

Alternative models (from sentence-transformers):

- `redis/langcache-embed-v1` (768 dims) - Otimised for semantic caching
- `all-mpnet-base-v2` (768 dims) - Higher quality
- `multi-qa-MiniLM-L6-cos-v1` (384 dims) - Questions & Answers (Q&A) optimised

## Setup and execution

This example requires the Redis Stack running on Docker. You can download and run the Redis Stack Docker container for free. The easiest way to do this is with [Docker Desktop](https://docs.docker.com/desktop/) for macOS and Microsoft Windows (or Linux), or with [Docker Engine](https://docs.docker.com/engine/install/) on Linux.

If you want to start the Redis Stack Docker container from command line, then you can run this command:

```bash
docker run -d --name redis -p 6379:6379 redis/redis-stack:latest
```

Just run the `main.py` script for this example. Check the output log for the results, which will look similar to the provided `EXAMPLE_output.txt` file. The script execution will also create a graph file called `cache_hitrate_f1_distance_threshold.png` with the optimisation results.

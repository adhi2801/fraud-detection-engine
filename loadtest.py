import asyncio
import aiohttp
import time
import json
import statistics
import random

API_URL = "http://localhost:8090/evaluate"
TOTAL_REQUESTS = 1000
CONCURRENT = 20

cards = [f"card-{i}" for i in range(100)]
merchants = [f"merch-{i:03d}" for i in range(5)]
countries = ["US", "US", "US", "US", "IN", "RU", "DE", "BR"]

def random_transaction():
    is_fraud = random.random() < 0.05
    return {
        "card_id": random.choice(cards),
        "amount_usd": random.uniform(500, 2000) if is_fraud else random.uniform(10, 200),
        "merchant_id": random.choice(merchants),
        "country_code": random.choice(countries) if is_fraud else "US"
    }

async def make_request(session, semaphore):
    async with semaphore:
        tx = random_transaction()
        start = time.perf_counter()
        try:
            async with session.post(API_URL, json=tx) as resp:
                data = await resp.json()
                elapsed = (time.perf_counter() - start) * 1000
                return elapsed, data.get("is_fraud", False)
        except Exception as e:
            return None, False

async def run_loadtest():
    print(f"Running load test: {TOTAL_REQUESTS} requests, {CONCURRENT} concurrent")
    print("-" * 50)

    semaphore = asyncio.Semaphore(CONCURRENT)
    latencies = []
    fraud_count = 0
    errors = 0

    start_time = time.time()

    async with aiohttp.ClientSession() as session:
        tasks = [make_request(session, semaphore) for _ in range(TOTAL_REQUESTS)]
        results = await asyncio.gather(*tasks)

    total_time = time.time() - start_time

    for latency, is_fraud in results:
        if latency is None:
            errors += 1
        else:
            latencies.append(latency)
            if is_fraud:
                fraud_count += 1

    if latencies:
        print(f"\n📊 LOAD TEST RESULTS")
        print(f"{'='*50}")
        print(f"Total requests:      {TOTAL_REQUESTS}")
        print(f"Successful:          {len(latencies)}")
        print(f"Errors:              {errors}")
        print(f"Total time:          {total_time:.2f}s")
        print(f"Throughput:          {len(latencies)/total_time:.0f} req/sec")
        print(f"Fraud detected:      {fraud_count} ({fraud_count/len(latencies)*100:.1f}%)")
        print(f"")
        print(f"⚡ LATENCY (milliseconds)")
        print(f"{'='*50}")
        print(f"P50 (median):        {statistics.median(latencies):.2f}ms")
        print(f"P95:                 {sorted(latencies)[int(len(latencies)*0.95)]:.2f}ms")
        print(f"P99:                 {sorted(latencies)[int(len(latencies)*0.99)]:.2f}ms")
        print(f"Min:                 {min(latencies):.2f}ms")
        print(f"Max:                 {max(latencies):.2f}ms")
        print(f"Mean:                {statistics.mean(latencies):.2f}ms")

if __name__ == "__main__":
    asyncio.run(run_loadtest())
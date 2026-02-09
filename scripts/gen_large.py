#!/usr/bin/env python3
import json, random, string, sys

random.seed(42)

def rand_str(lo, hi):
    return ''.join(random.choices(string.ascii_letters, k=random.randint(lo, hi)))

def rand_email():
    return rand_str(5, 10).lower() + "@example.com"

def make_record(i):
    return {
        "id": i,
        "name": rand_str(10, 30),
        "email": rand_email(),
        "score": round(random.uniform(0, 1000), 6),
        "active": random.choice([True, False]),
        "tags": [rand_str(3, 10).lower() for _ in range(random.randint(1, 5))],
        "address": {
            "city": rand_str(5, 20),
            "zip": f"{random.randint(10000, 99999)}",
            "lat": round(random.uniform(-90, 90), 8),
            "lng": round(random.uniform(-180, 180), 8),
        }
    }

n = int(sys.argv[1]) if len(sys.argv) > 1 else 500000
data = {"records": [make_record(i) for i in range(n)]}
json.dump(data, sys.stdout, separators=(',', ':'))
sys.stdout.write('\n')

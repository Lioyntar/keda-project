import json
import time
import uuid

import redis

r = redis.Redis(host="localhost", port=6379)
for _ in range(20):
    r.rpush("jobs", json.dumps({"id": str(uuid.uuid4()), "enqueued_at": time.time()}))
print("queue length:", r.llen("jobs"))
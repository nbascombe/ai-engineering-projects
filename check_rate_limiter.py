import time
from rag_chatbot.worker_tasks import wait_for_slot

t0 = time.time()
for i in range(6):
    wait_for_slot("test-model", limit=3, window=10, max_wait=30)
    print(i, round(time.time() - t0, 1))
import argparse
import math
import re
import statistics
import time
from collections import deque
from google.cloud import storage

HREF_PATTERN = re.compile(r'href=["\'](\d+)\.html["\']', re.IGNORECASE)

def download_with_retry(blob, max_retries=5):
    for attempt in range(max_retries):
        try:
            return blob.download_as_text(timeout=60)
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(1 + attempt)

def load_graph_from_gcs(bucket_name, prefix="files/"):
    print(f"Connecting to bucket '{bucket_name}'...")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    blobs = []
    for attempt in range(5):
        try:
            iterator = bucket.list_blobs(prefix=prefix, timeout=300, projection="noAcl")
            blobs = [b for b in iterator if b.name.endswith(".html")]
            break
        except Exception as e:
            print(f"  Listing retry {attempt + 1}/5 due to: {e}")
            time.sleep(2 ** attempt)

    n = len(blobs)
    if n == 0:
        raise ValueError(f"No .html files found in gs://{bucket_name}/{prefix}")
        
    print(f"Found {n} files. Beginning sequential single-threaded ingestion...\n")
    
    out_edges = [set() for _ in range(n)]
    in_edges = [set() for _ in range(n)]
    
    t_start = time.perf_counter()
    for idx, b in enumerate(blobs):
        src = int(b.name.split("/")[-1].replace(".html", ""))
        content = download_with_retry(b)
        
        for m in HREF_PATTERN.findall(content):
            target = int(m)
            if target < n and target != src:
                out_edges[src].add(target)
                in_edges[target].add(src)
                
        if (idx + 1) % 1000 == 0 or (idx + 1) == n:
            elapsed = time.perf_counter() - t_start
            rate = (idx + 1) / elapsed
            print(f"  [I/O Progress] Downloaded {idx + 1:5d}/{n} files ({rate:.1f} files/sec, {elapsed:.1f}s elapsed)")
            
    out_edges_list = [list(e) for e in out_edges]
    in_edges_list = [list(e) for e in in_edges]
    return n, out_edges_list, in_edges_list

def compute_degree_stats(degrees, label):
    data = sorted(degrees)
    n = len(data)
    avg_val = sum(data) / n
    med_val = statistics.median(data)
    min_val, max_val = data[0], data[-1]
    
    quintiles = [data[int(math.ceil(p * n)) - 1] for p in (0.2, 0.4, 0.6, 0.8)]
    
    print(f"\n{label} Distribution")
    print(f"Min: {min_val} | Max: {max_val} | Average: {avg_val:.2f} | Median: {med_val}")
    print(f"Quintiles (20%, 40%, 60%, 80%): {quintiles}")

def run_pagerank(n, out_edges, in_edges, damping=0.85, threshold=0.005):
    base_val = (1.0 - damping) / n
    pr = [1.0 / n] * n
    out_degrees = [len(e) for e in out_edges]
    
    iteration = 0
    print("\nIterative PageRank")
    while True:
        iteration += 1
        new_pr = [0.0] * n
        
        for i in range(n):
            incoming_sum = sum(pr[t] / out_degrees[t] for t in in_edges[i] if out_degrees[t] > 0)
            new_pr[i] = base_val + damping * incoming_sum
        
        old_sum = sum(pr)
        new_sum = sum(new_pr)
        delta = abs(new_sum - old_sum) / (old_sum if old_sum != 0 else 1.0)
        pr = new_pr
        
        print(f"  Iteration {iteration}: sum(PR) = {new_sum:.6f}, delta = {delta * 100:.4f}%")
        if delta <= threshold:
            print(f"PageRank converged in {iteration} iterations (delta <= 0.5%).")
            break
            
    return pr

def compute_closeness_centrality(n, out_edges):
    best_node = -1
    best_score = -1.0
    
    t_start = time.perf_counter()
    
    for src in range(n):
        dist = [-1] * n
        dist[src] = 0
        queue = deque([src])
        reachable_count = 0
        sum_distances = 0
        
        while queue:
            curr = queue.popleft()
            d_next = dist[curr] + 1
            for neighbor in out_edges[curr]:
                if dist[neighbor] == -1:
                    dist[neighbor] = d_next
                    reachable_count += 1
                    sum_distances += d_next
                    if reachable_count == n - 1:
                        queue.clear()
                        break
                    queue.append(neighbor)
                    
        if sum_distances > 0 and reachable_count > 0:
            score = (reachable_count / sum_distances) * (reachable_count / (n - 1))
        else:
            score = 0.0
            
        if score > best_score:
            best_score = score
            best_node = src
            
        if (src + 1) % 2000 == 0 or (src + 1) == n:
            elapsed = time.perf_counter() - t_start
            print(f"  [BFS Progress] Processed {src + 1:5d}/{n} nodes ({elapsed:.1f}s elapsed)...")
            
    return best_node, best_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", default="cs528-pagerank-eddielu-1790373540", help="GCS Bucket Name")
    parser.add_argument("--prefix", default="files/", help="Bucket subfolder prefix")
    args = parser.parse_args()

    t0 = time.perf_counter()
    n, out_edges, in_edges = load_graph_from_gcs(args.bucket, args.prefix)
    t_io = time.perf_counter() - t0

    compute_degree_stats([len(e) for e in in_edges], "Incoming Links")
    compute_degree_stats([len(e) for e in out_edges], "Outgoing Links")

    t1 = time.perf_counter()
    pr = run_pagerank(n, out_edges, in_edges)
    t_pr = time.perf_counter() - t1

    top_5 = sorted(enumerate(pr), key=lambda x: x[1], reverse=True)[:5]
    print("\nTop 5 Pages by PageRank:")
    for rank, (node, score) in enumerate(top_5, 1):
        print(f"  {rank}. Page {node}.html (Score: {score:.8f})")

    t2 = time.perf_counter()
    best_node, best_score = compute_closeness_centrality(n, out_edges)
    t_cc = time.perf_counter() - t2

    print(f"\nBest Closeness Centrality: Page {best_node}.html (Score: {best_score:.6f})")

    print(f"GCS Download & Ingestion: {t_io:8.2f} s")
    print(f"PageRank Computation (CPU): {t_pr:8.2f} s")
    print(f"Closeness Centrality (CPU): {t_cc:8.2f} s")
    print(f"Total Wall-Clock Time: {t_io + t_pr + t_cc:8.2f} s")

if __name__ == "__main__":
    main()
import asyncio, time, httpx, json, sys, os, subprocess, numpy as np
from pathlib import Path


async def resolve_fn(app_url):
    async with httpx.AsyncClient() as c:
        info = (await c.get(f"{app_url}/gradio_api/info", timeout=5)).json()
        api_name = list(info.get("named_endpoints", {}).keys())[0]
        config = (await c.get(f"{app_url}/config", timeout=5)).json()
        for dep in config.get("dependencies", []):
            if dep.get("api_name") == api_name.lstrip("/"):
                fn_index = dep.get("id", 0)
                components = {c["id"]: c for c in config.get("components", [])}
                data = []
                for cid in dep.get("inputs", []):
                    ct = components.get(cid, {}).get("type", "")
                    data.append(None if ct == "state" else "hello")
                return fn_index, data or ["hello"]
    return 0, ["hello"]


async def bench(app_url, name):
    fn_index, data = await resolve_fn(app_url)
    # Warmup
    async with httpx.AsyncClient() as c:
        for i in range(3):
            sh = f"wu_{name}_{i}"
            try:
                await c.post(
                    f"{app_url}/gradio_api/queue/join",
                    json={"data": data, "fn_index": fn_index, "session_hash": sh},
                    timeout=10,
                )
                async with c.stream(
                    "GET",
                    f"{app_url}/gradio_api/queue/data",
                    params={"session_hash": sh},
                    timeout=10,
                ) as s:
                    async for line in s.aiter_lines():
                        if "process_completed" in line:
                            break
            except:
                pass

    # Tier 100, 10 rounds, shared client
    latencies = []
    async with httpx.AsyncClient() as client:
        for rid in range(10):
            barrier = asyncio.Barrier(100)

            async def burst(uid, rid=rid, b=barrier):
                sh = f"b_{name}_{uid}_{rid}_{id(b)}"
                await b.wait()
                start = time.monotonic()
                try:
                    await client.post(
                        f"{app_url}/gradio_api/queue/join",
                        json={"data": data, "fn_index": fn_index, "session_hash": sh},
                        timeout=120,
                    )
                    async with client.stream(
                        "GET",
                        f"{app_url}/gradio_api/queue/data",
                        params={"session_hash": sh},
                        timeout=120,
                    ) as stream:
                        async for line in stream.aiter_lines():
                            if "process_completed" in line:
                                break
                    return (time.monotonic() - start) * 1000
                except:
                    return -1

            results = await asyncio.gather(*[burst(i) for i in range(100)])
            latencies.extend([r for r in results if r > 0])
    arr = np.array(latencies)
    print(
        f"{name:20s}: p50={np.percentile(arr, 50):.0f}ms p90={np.percentile(arr, 90):.0f}ms n={len(latencies)}",
        flush=True,
    )


async def main():
    apps = [
        ("scripts/benchmark/apps/echo_text.py", 7891),
        ("scripts/benchmark/apps/file_heavy.py", 7892),
        ("scripts/benchmark/apps/stateful_counter.py", 7893),
        ("scripts/benchmark/apps/streaming_chat.py", 7894),
    ]
    procs = []
    for app_path, port in apps:
        env = os.environ.copy()
        env["GRADIO_PROFILING"] = "1"
        env["GRADIO_SERVER_PORT"] = str(port)
        p = subprocess.Popen(
            [sys.executable, app_path],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        procs.append(p)

    for _, port in apps:
        for _ in range(10):
            try:
                r = httpx.get(f"http://127.0.0.1:{port}/gradio_api/info", timeout=2)
                if r.status_code == 200:
                    break
            except:
                pass
            time.sleep(0.5)
    print("All servers ready", flush=True)

    await asyncio.gather(
        *[bench(f"http://127.0.0.1:{port}", Path(app).stem) for app, port in apps]
    )

    for p in procs:
        p.terminate()
        p.wait(timeout=5)


asyncio.run(main())

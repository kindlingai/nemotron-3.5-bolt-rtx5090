# Fire N streams at the same instant; report per-stream decode, aggregate, TTFT.
import asyncio,json,sys,time,aiohttp
URL=sys.argv[1] if len(sys.argv)>1 else "http://localhost:8000/v1/chat/completions"
N=int(sys.argv[2]) if len(sys.argv)>2 else 4; TOK=512
async def one(s,i,go):
    b={"model":"nemotron-3.5-bolt","messages":[{"role":"user","content":f"Write a long story about lighthouse #{i}."}],
       "max_tokens":TOK,"ignore_eos":True,"stream":True,"stream_options":{"include_usage":True},
       "chat_template_kwargs":{"enable_thinking":False}}
    await go.wait(); t0=time.time(); first=None; n=0
    async with s.post(URL,json=b) as r:
        async for line in r.content:
            line=line.decode().strip()
            if not line.startswith("data: ") or line.endswith("[DONE]"): continue
            d=json.loads(line[6:])
            if d.get("usage"): n=d["usage"]["completion_tokens"]
            if first is None and any((c.get("delta") or {}).get("content") for c in d.get("choices",[])): first=time.time()
    return t0,first,time.time(),n
async def main():
    go=asyncio.Event()
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=900)) as s:
        ts=[asyncio.create_task(one(s,i,go)) for i in range(N)]
        await asyncio.sleep(0.2); go.set(); rs=await asyncio.gather(*ts)
    t0=min(r[0] for r in rs); t1=max(r[2] for r in rs); tot=sum(r[3] for r in rs)
    per=[r[3]/(r[2]-r[1]) for r in rs]; ttft=[r[1]-r[0] for r in rs]
    print(f"S{N} aggregate={tot/(t1-t0):.1f} tok/s per-stream mean={sum(per)/N:.1f} tok/s ttft={min(ttft):.2f}-{max(ttft):.2f}s tokens={tot}")
asyncio.run(main())

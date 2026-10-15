import asyncio,aiohttp,random,sys,time
import os
URL=os.environ.get("URL","http://localhost:8000/v1/chat/completions"); N=int(sys.argv[1]); L=int(sys.argv[2])
w="alpha beta gamma delta river stone cloud paper metal glass".split()
async def one(s,i):
    random.seed(1000+i); msg=" ".join(random.choice(w) for _ in range(L))+f"\nRequest {i}: say the word ok."
    b={"model":"nemotron-3.5-bolt","messages":[{"role":"user","content":msg}],"max_tokens":256,"ignore_eos":True,"chat_template_kwargs":{"enable_thinking":False}}
    t=time.time()
    async with s.post(URL,json=b) as r: d=await r.json()
    return r.status, d.get("usage",{}).get("prompt_tokens"), d.get("usage",{}).get("completion_tokens"), time.time()-t
async def main():
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1800)) as s:
        t=time.time(); rs=await asyncio.gather(*[one(s,i) for i in range(N)])
    print(f"STRESS N={N} L={L} wall={time.time()-t:.1f}s statuses={[r[0] for r in rs]} completion={[r[2] for r in rs]} times={[round(r[3]) for r in rs]}")
asyncio.run(main())

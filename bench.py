import json,time,sys,random,urllib.request,concurrent.futures as cf
U="http://localhost:8000/v1/chat/completions"; M="nemotron-3.5-bolt"
def req(msgs,max_tokens,stream=True,think=False,IE=True):
    b={"model":M,"messages":msgs,"max_tokens":max_tokens,"temperature":1.0,"top_p":0.95,"stream":stream,
       "chat_template_kwargs":{"enable_thinking":think},"ignore_eos":IE,"stream_options":{"include_usage":True}}
    r=urllib.request.Request(U,json.dumps(b).encode(),{"Content-Type":"application/json"})
    t0=time.time();first=None;usage=None;text=""
    with urllib.request.urlopen(r,timeout=900) as f:
        for line in f:
            line=line.decode().strip()
            if not line.startswith("data: ") or line=="data: [DONE]": continue
            d=json.loads(line[6:])
            if d.get("usage"): usage=d["usage"]
            for c in d.get("choices",[]):
                dl=c.get("delta",{}); s=(dl.get("content") or "")+(dl.get("reasoning_content") or dl.get("reasoning") or "")
                if s and first is None: first=time.time()
                text+=s
    return t0,first,time.time(),usage,text
def filler(n): 
    random.seed(n+time.time()); w="alpha beta gamma delta river stone cloud paper metal glass".split()
    return " ".join(random.choice(w) for _ in range(n-40))
mode=sys.argv[1]
if mode=="smoke":
    t0,f,t1,u,txt=req([{"role":"user","content":"What is 17*23? Answer with the number only."}],64,think=False,IE=False)
    print("SMOKE", repr(txt.strip()[:80]), "PASS" if "391" in txt else "FAIL")
elif mode=="prefill":
    for n in (8192,32768):
        t0,f,t1,u,_=req([{"role":"user","content":filler(n)+"\nSay ok."}],1)
        p=u["prompt_tokens"]; print(f"PREFILL ctx={p} ttft={f-t0 if f else t1-t0:.2f}s rate={p/((f or t1)-t0):.0f} t/s")
elif mode=="decode":
    S=int(sys.argv[2]); N=512
    def one(i): return req([{"role":"user","content":f"Write a long detailed story about a lighthouse keeper #{i}."}],N)
    with cf.ThreadPoolExecutor(S) as ex: rs=list(ex.map(one,range(S)))
    per=[r[3]["completion_tokens"]/(r[2]-r[1]) for r in rs]; wall=max(r[2] for r in rs)-min(r[0] for r in rs)
    print(f"DECODE S{S} per-stream={sum(per)/S:.1f} tok/s aggregate={sum(r[3]['completion_tokens'] for r in rs)/wall:.1f} tok/s")

import json,os,random,sys,time,urllib.request
URL=os.environ.get("URL","http://localhost:8000/v1/chat/completions")
N=int(sys.argv[1]) if len(sys.argv)>1 else 200000
random.seed(int(sys.argv[2]) if len(sys.argv)>2 else 7); w="alpha beta gamma delta river stone cloud paper metal glass".split()
words=[random.choice(w) for _ in range(int(N*0.98))]
code="KX-%d"%random.randint(100000,999999)
words.insert(len(words)//2, f". The secret vault code is {code}. ")
msg=" ".join(words)+"\n\nWhat is the secret vault code? Reply with the code only."
b={"model":"nemotron-3.5-bolt","messages":[{"role":"user","content":msg}],"max_tokens":32,"temperature":0,
   "chat_template_kwargs":{"enable_thinking":False}}
t=time.time()
r=json.load(urllib.request.urlopen(urllib.request.Request(URL,json.dumps(b).encode(),{"Content-Type":"application/json"}),timeout=1800))
dt=time.time()-t; p=r["usage"]["prompt_tokens"]; a=r["choices"][0]["message"]["content"] or ""
print(f"NEEDLE prompt={p} time={dt:.1f}s rate={p/dt:.0f} t/s answer={a.strip()[:40]!r} {'PASS' if code in a else 'FAIL'} (code {code})")

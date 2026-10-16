"""Ephemeral signed URLs come from a CI secret; never written to files or logs."""
import concurrent.futures
import hashlib
import json
import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
import requests

ROOT=Path(__file__).resolve().parents[1]

def open_remote(slot,run,name):
    mapping=json.loads(os.environ['CHECKPOINT_URLS'])
    assert 'manifest.ocdbt' in mapping
    cache=ROOT/'checkpoint-cache'/str(slot);cache.mkdir(parents=True,exist_ok=True)
    session=requests.Session()
    session.mount('https://',requests.adapters.HTTPAdapter(pool_connections=24,pool_maxsize=24,pool_block=True))
    gate=threading.Semaphore(24);sizes={};locks={};lock=threading.Lock()
    def fetch(path,start,end):
        key=hashlib.sha256(f'{path}:{start}:{end}'.encode()).hexdigest();file=cache/key
        with lock:resource=locks.setdefault(key,threading.Lock())
        with resource:
            if file.exists() and file.stat().st_size==end-start+1:return file.read_bytes()
            with gate:
                for attempt in range(5):
                    try:
                        r=session.get(mapping[path],headers={'Range':f'bytes={start}-{end}'},timeout=(20,90))
                        assert r.status_code==206 and len(r.content)==end-start+1
                        assert r.headers['Content-Range'].startswith(f'bytes {start}-{end}/')
                        file.write_bytes(r.content);return r.content
                    except Exception:
                        if attempt==4:raise RuntimeError(f'Checkpoint range unavailable: {path}, offset {start}') from None
                        time.sleep(2)
    def size(path):
        if path not in sizes:
            r=session.get(mapping[path],headers={'Range':'bytes=0-0'},timeout=(20,60))
            assert r.status_code==206
            sizes[path]=int(r.headers['Content-Range'].split('/')[-1])
        return sizes[path]
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_HEAD(self):self.serve(False)
        def do_GET(self):self.serve(True)
        def serve(self,body):
            path=self.path.split('?',1)[0].lstrip('/')
            if path not in mapping:self.send_error(404);return
            try:
                total=size(path);rg=self.headers.get('Range')
                if rg:
                    match=re.fullmatch(r'bytes=(\d+)-(\d*)',rg)
                    if not match:self.send_error(416);return
                    start=int(match[1]);end=min(total-1,int(match[2]) if match[2] else total-1)
                else:start,end=0,total-1
                self.send_response(206 if rg else 200);self.send_header('Content-Length',str(end-start+1))
                self.send_header('Accept-Ranges','bytes')
                if rg:self.send_header('Content-Range',f'bytes {start}-{end}/{total}')
                self.end_headers()
                if not body:return
                stride=1024*1024
                spans=[(s,min(total-1,s+stride-1)) for s in range(start//stride*stride,end+1,stride)]
                with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
                    futures=[pool.submit(fetch,path,a,b) for a,b in spans]
                    for (a,b),future in zip(spans,futures):
                        raw=future.result();self.wfile.write(raw[max(0,start-a):min(len(raw),end-a+1)])
            except (BrokenPipeError,ConnectionResetError):pass
            except Exception:
                self.close_connection=True
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    return {'driver':'ocdbt','base':{'driver':'http','base_url':f'http://127.0.0.1:{server.server_port}/'}},server

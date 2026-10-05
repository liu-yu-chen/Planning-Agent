"""Owner-operated read-only retrieval gateway behind an HTTPS reverse proxy."""
import hmac
import json
import os
import time
from collections import defaultdict
from datetime import date
from http.server import BaseHTTPRequestHandler,HTTPServer
from .retriever import Retriever
from .live import OpenAlex


def serve(settings,host='127.0.0.1',port=8765):
    if host not in ('127.0.0.1','localhost','::1'):
        raise ValueError('This gateway binds to loopback only. Publish through an authenticated HTTPS reverse proxy.')
    env=settings.get('service_token_env','PLANNING_SERVICE_TOKEN')
    token=os.environ.get(env)
    if not token or len(token)<24: raise ValueError(f'Set {env} to an access token of at least 24 characters before serving.')
    live=OpenAlex(settings.get('openalex_cache'),settings.get('openalex_key_env','OPENALEX_API_KEY'))
    minute_calls=defaultdict(int);day_calls=defaultdict(int)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,fmt,*args): pass  # Do not record credentials, queries or source excerpts.
        def respond(self,status,value):
            data=json.dumps(value,ensure_ascii=False).encode('utf-8')
            self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8')
            self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
        def authorize(self):
            if not hmac.compare_digest(self.headers.get('Authorization',''),'Bearer '+token):
                self.respond(401,{'error':'Access denied'});return False
            minute=int(time.time()//60);today=date.today().isoformat()
            # Global bounds for this owner process. Daily cache persists across restarts.
            if minute_calls[minute]>=30 or day_calls[today]>=settings.get('service_daily_request_cap',1000):
                self.respond(429,{'error':'Owner gateway request allowance exceeded'});return False
            for old in list(minute_calls):
                if old<minute-1: del minute_calls[old]
            for old in list(day_calls):
                if old!=today: del day_calls[old]
            minute_calls[minute]+=1;day_calls[today]+=1;return True
        def do_GET(self):
            if self.path=='/health': self.respond(200,{'status':'available','service':'planning-review','version':'0.1.0'});return
            if not self.authorize(): return
            if self.path!='/scope': self.respond(404,{'error':'Unknown endpoint'});return
            rag=None
            try:
                rag=Retriever(settings['database'])
                self.respond(200,{'corpus':rag.scope(),'retrieval_mode':'lexical_fts5',
                    'graph_available':rag.graph_available,'live_source':'OpenAlex','live_from_year':2026})
            except Exception: self.respond(503,{'error':'Corpus unavailable or incomplete on the owner host'})
            finally:
                if rag: rag.close()
        def do_POST(self):
            if not self.authorize(): return
            rag=None
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=16000: self.respond(413,{'error':'Request must contain 1-16000 bytes'});return
                data=json.loads(self.rfile.read(length))
                if not isinstance(data,dict): raise ValueError('Request must be a JSON object.')
                if self.path=='/recent':
                    result=live.search(data['query'],data.get('from_year',2026),data.get('to_year'),
                        min(int(data.get('limit',20)),100),data.get('minimum_citations',1))
                elif self.path=='/search':
                    q=data['question']
                    if not isinstance(q,dict) or any(not isinstance(q.get(k),str) or not 0<len(q[k])<=1500 for k in ('id','question','search_query')):
                        raise ValueError('Missing or oversized question fields.')
                    if len(str(q.get('fts_expression') or ''))>2000: raise ValueError('FTS expression is too long.')
                    lo=q.get('from_year');hi=q.get('to_year')
                    for y in (lo,hi):
                        if y is not None and not 1900<=int(y)<=date.today().year: raise ValueError('Invalid year filter.')
                    rag=Retriever(settings['database'],lo,hi)
                    result=rag.retrieve(q,max(1,min(int(data.get('top_k',4)),25)),max(10,min(int(data.get('candidate_k',40)),100)))
                elif self.path=='/paper':
                    from .retriever import normalize_doi
                    doi=normalize_doi(data.get('doi'))
                    if not doi: raise ValueError('A valid DOI is required.')
                    rag=Retriever(settings['database']);result=rag.paper(doi)
                    if not result: result=live.paper(doi)
                    result={'paper':result}
                else: self.respond(404,{'error':'Unknown endpoint'});return
                self.respond(200,result)
            except (KeyError,TypeError,ValueError): self.respond(400,{'error':'Invalid query parameters'})
            except RuntimeError as exc: self.respond(503,{'error':str(exc)})
            except Exception: self.respond(503,{'error':'Retrieval failed; inspect the owner configuration'})
            finally:
                if rag: rag.close()
    server=HTTPServer((host,int(port)),Handler);server.timeout=30
    print(f'Owner gateway listening at http://{host}:{port}; remote clients require an HTTPS reverse proxy.',flush=True)
    try: server.serve_forever(poll_interval=.5)
    except KeyboardInterrupt: pass
    finally: server.server_close()

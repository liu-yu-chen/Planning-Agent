"""Interchangeable local, shared-service and live literature access."""
import json
import os
import urllib.parse
import urllib.request
from .retriever import Retriever
from .live import OpenAlex


class Remote:
    def __init__(self,url,token_env='PLANNING_SERVICE_TOKEN'):
        parts=urllib.parse.urlparse(url)
        if parts.scheme not in ('https','http') or not parts.hostname or parts.username or parts.password or parts.query or parts.fragment:
            raise ValueError('Use a service URL without embedded credentials or query parameters.')
        if parts.scheme=='http' and parts.hostname not in ('localhost','127.0.0.1','::1'):
            raise ValueError('Remote shared services require HTTPS; local loopback may use HTTP.')
        self.url=url.rstrip('/');self.token_env=token_env
    def call(self,path,payload=None):
        headers={'Accept':'application/json','Content-Type':'application/json'}
        token=os.environ.get(self.token_env)
        if token: headers['Authorization']='Bearer '+token
        req=urllib.request.Request(self.url+path,data=json.dumps(payload).encode() if payload is not None else None,headers=headers)
        try:
            with urllib.request.urlopen(req,timeout=90) as response: return json.load(response)
        except Exception:
            raise RuntimeError('Shared retrieval service unavailable or access denied. Check its URL, token environment variable and owner availability.') from None
    def scope(self): return self.call('/scope')
    def retrieve(self,question,top_k=4,candidate_k=40): return self.call('/search',{'question':question,'top_k':top_k,'candidate_k':candidate_k})
    def close(self): pass


class Backend:
    def __init__(self,settings,cancel=None):
        self.settings=settings;self.local=None
        kind=settings.get('backend','local')
        if kind=='local': self.local=Retriever(settings.get('database',''),settings.get('from_year'),settings.get('to_year'))
        elif kind=='remote': self.local=Remote(settings.get('service_url',''),settings.get('service_token_env','PLANNING_SERVICE_TOKEN'))
        elif kind!='live': raise ValueError('Backend must be local, remote or live.')
        self.live=OpenAlex(settings.get('openalex_cache'),settings.get('openalex_key_env','OPENALEX_API_KEY'),cancel) if settings.get('live_enabled') or kind=='live' else None
    def close(self):
        if self.local: self.local.close()
    def scope(self):
        return {'backend':self.settings.get('backend','local'),
            'corpus':self.local.scope() if self.local else None,
            'from_year':self.settings.get('from_year'),'to_year':self.settings.get('to_year'),
            'live_enabled':bool(self.live),'live_from_year':self.settings.get('live_from_year',2026),
            'live_citation_floor':self.settings.get('minimum_citations',1),
            'retrieval_note':'Local: lexical FTS5 plus bibliographic graph; live: date-filtered OpenAlex search, without graph expansion. Shared service publishes its own scope.'}
    def retrieve(self,question,top_k=4,candidate_k=40):
        q=question|{'from_year':self.settings.get('from_year'),'to_year':self.settings.get('to_year')}
        event=self.local.retrieve(q,top_k,candidate_k) if self.local else {
            'query_id':q['id'],'question':q['question'],'search_query':q['search_query'],
            'modes':{'graph':[],'ordinary':[]},'graph_available':False,'shared_seed_dois':[],'retrieval_mode':'openalex_live'}
        if self.live:
            recent=self.live.search(q['search_query'],self.settings.get('live_from_year',2026),
                self.settings.get('to_year'),top_k,self.settings.get('minimum_citations',1))
            event['live_discovery']=recent
            for mode in ('graph','ordinary'):
                seen={p['doi'] for p in event['modes'][mode]}
                for p in recent['papers']:
                    if p['abstract'] and p['doi'] not in seen: event['modes'][mode].append(p);seen.add(p['doi'])
        return event

"""Bounded, cached OpenAlex discovery. Credentials are environment variables."""
from __future__ import annotations
import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path
from .retriever import normalize_doi,utc_now,write_json

PLANNING_CONTEXT='(urban OR city OR cities OR rural OR village OR regional OR spatial OR neighbourhood OR neighborhood OR planning)'


class OpenAlex:
    def __init__(self,cache=None,key_env='OPENALEX_API_KEY',cancel=None):
        self.cache=Path(cache or Path.home()/'.planning-review-agent'/'openalex-cache')
        self.key_env=key_env;self.cancel=cancel or threading.Event()
        self.lock=threading.Lock();self.last_call=0
        self.opener=urllib.request.build_opener()

    def _request(self,params):
        # Daily cache keys include all query filters, never credentials.
        identity=json.dumps(params,sort_keys=True)+date.today().isoformat()
        path=self.cache/(hashlib.sha256(identity.encode()).hexdigest()+'.json')
        if path.exists():
            saved=json.loads(path.read_text(encoding='utf-8'));saved['cache_hit']=True;return saved
        url='https://api.openalex.org/works?'+urllib.parse.urlencode(params)
        headers={'Accept':'application/json','User-Agent':'PlanningReviewAgent/0.1.0'}
        key=os.environ.get(self.key_env)
        if key: headers['Authorization']='Bearer '+key
        for attempt in range(4):
            if self.cancel.is_set(): raise InterruptedError('OpenAlex discovery cancelled.')
            with self.lock:
                if self.cancel.wait(max(0,1-(time.monotonic()-self.last_call))): raise InterruptedError('Cancelled.')
                self.last_call=time.monotonic()
                try:
                    with self.opener.open(urllib.request.Request(url,headers=headers),timeout=30) as response:
                        payload=json.load(response)
                        budget={name:response.headers.get(name) for name in ('X-RateLimit-Limit','X-RateLimit-Remaining','X-RateLimit-Credits-Used','X-RateLimit-Reset')}
                    value={'data':payload,'retrieved_at_utc':utc_now(),'query_parameters':params,'budget':budget,'cache_hit':False}
                    write_json(path,value);return value
                except urllib.error.HTTPError as exc:
                    if exc.code not in (429,500,502,503,504):
                        raise RuntimeError(f'OpenAlex HTTP {exc.code}. Check filters and account authorization.') from None
                    delay=2**attempt*2
                    if exc.code==429:
                        remaining=exc.headers.get('X-RateLimit-Remaining')
                        if remaining and remaining.strip()=='0':
                            raise RuntimeError('OpenAlex daily budget exhausted; cached results remain available. Retry after the reset.') from None
                        try: delay=max(delay,float(exc.headers.get('Retry-After','0')))
                        except ValueError: pass
                    if delay>60: raise RuntimeError('OpenAlex requires a longer wait. Retry after the recorded rate-limit reset.') from None
                except (urllib.error.URLError,TimeoutError,OSError): delay=2**attempt*2
            if attempt==3: break
            if self.cancel.wait(delay): raise InterruptedError('Cancelled.')
        raise RuntimeError('OpenAlex could not return data after four bounded attempts.')

    @staticmethod
    def record(work):
        doi=normalize_doi(work.get('doi'))
        if not doi: return None
        inverted=work.get('abstract_inverted_index') or {}
        positions={p:w for w,ps in inverted.items() for p in ps}
        abstract=' '.join(positions[p] for p in sorted(positions))
        authors=[(a.get('author') or {}).get('display_name') for a in work.get('authorships',[])]
        return {'doc_id':doi,'doi':doi,'title':work.get('display_name') or work.get('title'),
            'abstract':abstract,'year':work.get('publication_year'),'publication_date':work.get('publication_date'),
            'journal':((work.get('primary_location') or {}).get('source') or {}).get('display_name'),
            'authors':[a for a in authors if a],'language':work.get('language'),
            'document_type':work.get('type'),'openalex_id':work.get('id'),
            'cited_by_count':work.get('cited_by_count'),'topics':work.get('topics',[]),
            'keywords':work.get('keywords',[]),'referenced_works':work.get('referenced_works',[]),
            'open_access':work.get('open_access'),'graph_paths':[],'source':'openalex_live'}

    def search(self,query,from_year=2026,to_year=None,limit=20,minimum_citations=1,document_types=None):
        query=str(query).strip()
        if not query or len(query)>1500: raise ValueError('Use a concise English query of 1-1500 characters.')
        if any('\u4e00'<=ch<='\u9fff' for ch in query): raise ValueError('Translate the search query into English in your LLM host first.')
        start=int(from_year);end=min(int(to_year or date.today().year),date.today().year)
        if not 1900<=start<=end: raise ValueError('Invalid publication-year range.')
        limit=max(1,min(int(limit),100));minimum=max(0,int(minimum_citations))
        types=document_types or ['article','review']
        if any(t not in ('article','review','preprint','book-chapter','proceedings-article') for t in types):
            raise ValueError('Unsupported document type.')
        end_date=min(date(end,12,31),date.today()).isoformat()
        filters=f'from_publication_date:{start}-01-01,to_publication_date:{end_date},language:en,type:{"|".join(types)},has_doi:true,cited_by_count:>{minimum-1}'
        value=self._request({'search.title_and_abstract':f'({query}) AND {PLANNING_CONTEXT}',
            'filter':filters,'per_page':100})
        rows=[]
        for work in value['data'].get('results',[]):
            p=self.record(work)
            if p:
                p['retrieved_at_utc']=value['retrieved_at_utc'];rows.append(p)
                if len(rows)>=limit: break
        return {'papers':rows,'total_hits':value['data'].get('meta',{}).get('count'),
            'source':'openalex_live','query':query,'from_year':start,'to_year':end,
            'minimum_citations':minimum,'document_types':types,'query_parameters':value['query_parameters'],
            'retrieved_at_utc':value['retrieved_at_utc'],'cache_hit':value['cache_hit'],'budget':value['budget'],
            'graph_available':False,'full_text_read':False,
            'coverage_note':'One relevance-ranked page of up to 100 records; not exhaustive. Missing abstracts remain empty. Indexing delays and the citation threshold may exclude recent papers. Read every retained title and abstract.'}

    def paper(self,doi):
        doi=normalize_doi(doi)
        if not doi: raise ValueError('A valid DOI is required.')
        value=self._request({'filter':'doi:https://doi.org/'+doi,'per_page':1})
        results=value['data'].get('results') or []
        if not results: return None
        p=self.record(results[0])
        if p: p['retrieved_at_utc']=value['retrieved_at_utc']
        return p

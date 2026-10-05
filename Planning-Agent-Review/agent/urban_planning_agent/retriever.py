"""Streaming metadata import and bounded lexical/graph retrieval (stdlib only)."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import re
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    with temp.open('w',encoding='utf-8') as stream:
        stream.write(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
        stream.flush();os.fsync(stream.fileno())
    temp.replace(path)


def normalize_doi(value):
    value=str(value or '').strip().lower()
    value=re.sub(r'^(?:https?://(?:dx\.)?doi\.org/|doi\s*:\s*)','',value)
    return value if re.match(r'^10\.\d{4,9}/\S+$',value) else None


def normalize_record(raw):
    meta=raw.get('metadata') or raw
    title=raw.get('title') or raw.get('title_raw') or meta.get('title') or meta.get('display_name')
    abstract=raw.get('abstract') or raw.get('abstract_raw') or meta.get('abstract')
    if not abstract and meta.get('abstract_inverted_index'):
        positions={pos:word for word,indices in meta['abstract_inverted_index'].items() for pos in indices}
        abstract=' '.join(positions[pos] for pos in sorted(positions))
    doi=normalize_doi(meta.get('doi') or (meta.get('ids') or {}).get('doi'))
    if not title or not abstract or not doi:
        return None,'Missing title, abstract or valid DOI'
    language=str(meta.get('language') or 'unknown').lower()
    if language not in ('en','english','unknown'):
        return None,'Explicit non-English language'
    year=meta.get('year') or meta.get('publication_year')
    try: year=int(year) if year else None
    except (ValueError,TypeError): year=None
    clean={'doi':doi,'year':year,'journal':meta.get('journal') or ((meta.get('primary_location') or {}).get('source') or {}).get('display_name'),
        'language':language,'authors_wos':meta.get('authors_wos') or meta.get('authors'),
        'cited_by_count':meta.get('cited_by_count',meta.get('citation_count')),
        'openalex_id':meta.get('openalex_id') or meta.get('id'),
        'referenced_works':meta.get('openalex_referenced_works',meta.get('referenced_works',[])),
        'openalex_topics':meta.get('openalex_topics',meta.get('topics',[])),
        'openalex_keywords':meta.get('openalex_keywords',meta.get('keywords',[])),
        'openalex_authorships':meta.get('openalex_authorships',meta.get('authorships',[])),
        'original_record':raw}
    return {'doc_id':doi,'title':str(title),'abstract':str(abstract),'metadata':clean},None


def import_jsonl(source,target,progress=lambda *a,**k:None,cancel=None):
    """Create a new index; DOI-deduplicate in SQLite without loading the corpus."""
    source=Path(source).resolve();target=Path(target).resolve()
    if target.exists(): raise ValueError('Choose a new index filename; existing indexes are never overwritten.')
    if source==target: raise ValueError('Input and output must be different files.')
    target.parent.mkdir(parents=True,exist_ok=True)
    db=sqlite3.connect(target)
    db.executescript('''
        CREATE TABLE documents(doc_id TEXT PRIMARY KEY,title TEXT NOT NULL,abstract TEXT NOT NULL,openalex_id TEXT,metadata_json TEXT NOT NULL);
        CREATE TABLE chunks(chunk_id INTEGER PRIMARY KEY,doc_id TEXT NOT NULL,text TEXT NOT NULL);
        CREATE VIRTUAL TABLE chunk_fts USING fts5(doc_id UNINDEXED,chunk_id UNINDEXED,text);
        CREATE TABLE citation_eligibility(doc_id TEXT PRIMARY KEY,eligible INTEGER NOT NULL);
        CREATE TABLE graph_memberships(doc_id TEXT,relation TEXT,entity_key TEXT,entity_label TEXT);
        CREATE TABLE citation_edges(source_doc_id TEXT,target_doc_id TEXT,PRIMARY KEY(source_doc_id,target_doc_id));
        CREATE TABLE pending_references(source_doc_id TEXT,target_key TEXT);
        CREATE TABLE import_provenance(doc_id TEXT,source_line INTEGER,original_json TEXT);
        CREATE TABLE import_state(name TEXT PRIMARY KEY,value TEXT);
        INSERT INTO import_state VALUES('status','building');
        CREATE INDEX graph_entity ON graph_memberships(relation,entity_key);
        CREATE INDEX graph_doc ON graph_memberships(doc_id);
        CREATE INDEX chunk_doc ON chunks(doc_id);
        CREATE INDEX document_openalex ON documents(openalex_id);
    ''')
    counts=defaultdict(int);size=source.stat().st_size;consumed=0
    def entity(doi,relation,key,label):
        if key: db.execute('INSERT INTO graph_memberships VALUES(?,?,?,?)',(doi,relation,str(key),str(label or key)))
    try:
        with source.open('rb') as stream:
            for line in stream:
                if cancel and cancel.is_set(): raise InterruptedError('Import cancelled; partial index is marked incomplete. Choose a new target to restart import.')
                consumed+=len(line)
                if not line.strip(): continue
                counts['raw_records']+=1
                try: p,reason=normalize_record(json.loads(line))
                except (ValueError,TypeError,AttributeError): p,reason=None,'Malformed record'
                if p is None:
                    counts[reason]+=1;continue
                m=p['metadata'];doi=p['doc_id']
                if re.fullmatch(r'W\d+',str(m.get('openalex_id') or '')): m['openalex_id']='https://openalex.org/'+m['openalex_id']
                m['referenced_works']=['https://openalex.org/'+str(r) if re.fullmatch(r'W\d+',str(r)) else r for r in m.get('referenced_works') or []]
                db.execute('INSERT INTO import_provenance VALUES(?,?,?)',(doi,counts['raw_records'],line.decode('utf-8-sig').strip()))
                cur=db.execute('INSERT OR IGNORE INTO documents VALUES(?,?,?,?,?)',
                    (doi,p['title'],p['abstract'],m.get('openalex_id'),json.dumps(m,ensure_ascii=False)))
                if not cur.rowcount:
                    counts['duplicate_doi']+=1;continue
                counts['unique_records']+=1
                text=p['title']+'\n'+p['abstract']
                chunk=db.execute('INSERT INTO chunks(doc_id,text) VALUES(?,?)',(doi,text)).lastrowid
                db.execute('INSERT INTO chunk_fts(doc_id,chunk_id,text) VALUES(?,?,?)',(doi,chunk,text))
                db.execute('INSERT INTO citation_eligibility VALUES(?,1)',(doi,))
                for name,relation in (('openalex_topics','topic'),('openalex_keywords','keyword')):
                    for item in m.get(name) or []:
                        if isinstance(item,dict): entity(doi,relation,item.get('id') or item.get('name') or item.get('display_name'),item.get('name') or item.get('display_name'))
                for auth in m.get('openalex_authorships') or []:
                    if not isinstance(auth,dict): continue
                    author=auth.get('author') or {}
                    if isinstance(author,dict): entity(doi,'author',author.get('id') or author.get('display_name'),author.get('display_name'))
                    elif author: entity(doi,'author',author,author)
                    for inst in auth.get('institutions') or []:
                        if isinstance(inst,dict): entity(doi,'institution',inst.get('id') or inst.get('name') or inst.get('display_name'),inst.get('name') or inst.get('display_name'))
                for ref in m.get('referenced_works') or []:
                    db.execute('INSERT INTO pending_references VALUES(?,?)',(doi,str(ref)))
                if counts['unique_records']%1000==0:
                    db.commit();progress('Importing metadata',consumed,size)
        db.execute('INSERT OR IGNORE INTO citation_edges SELECT p.source_doc_id,d.doc_id FROM pending_references p JOIN documents d ON d.openalex_id=p.target_key OR d.doc_id=p.target_key')
        db.executescript('CREATE INDEX citation_target ON citation_edges(target_doc_id); DROP TABLE pending_references;')
        db.execute("UPDATE import_state SET value='complete' WHERE name='status'")
        db.commit()
    except BaseException:
        db.commit();raise
    finally: db.close()
    manifest={'created_at_utc':utc_now(),'import_counts':dict(counts),'database_name':target.name,
        'eligibility':'DOI, nonempty title/abstract, explicit non-English excluded; missing language retained as unknown; no citation-age filter applied to this imported corpus',
        'retrieval':'Lexical FTS5 and graph expansion; no embeddings created','import_complete':True}
    write_json(target.with_suffix('.manifest.json'),manifest)
    progress('Import complete',size,size)
    return manifest


class Retriever:
    def __init__(self,path,from_year=None,to_year=None):
        self.from_year=int(from_year) if from_year else None
        self.to_year=int(to_year) if to_year else None
        self.path=Path(path).resolve()
        if not self.path.is_file(): raise FileNotFoundError('Select an existing literature SQLite index or import a JSONL corpus.')
        self.db=sqlite3.connect(self.path.as_uri()+'?mode=ro',uri=True)
        self.tables={r[0] for r in self.db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not {'documents','chunks','chunk_fts'}<=self.tables:
            self.db.close();raise ValueError('The index needs documents, chunks and FTS5 tables.')
        if 'import_state' in self.tables:
            state=self.db.execute("SELECT value FROM import_state WHERE name='status'").fetchone()
            if not state or state[0]!='complete':
                self.db.close();raise ValueError('This imported index is incomplete. Import again into a new target filename.')
        self.filtered='citation_eligibility' in self.tables
        self.graph_available={'graph_memberships','citation_edges'}<=self.tables

    def close(self): self.db.close()

    def scope(self):
        for path in (self.path.parent/'citation_filter_manifest.json',self.path.with_suffix('.manifest.json')):
            if path.exists(): return json.loads(path.read_text(encoding='utf-8'))
        return {'database_name':self.path.name,'eligibility_table_present':self.filtered,
                'eligibility_details':'No matching sidecar manifest; eligibility enforced if present.'}

    def paper(self,doi):
        eligibility=' JOIN citation_eligibility e ON e.doc_id=d.doc_id AND e.eligible=1' if self.filtered else ''
        row=self.db.execute('SELECT d.doc_id,d.title,d.abstract,d.metadata_json FROM documents d'+eligibility+' WHERE d.doc_id=?',(doi,)).fetchone()
        if not row: return None
        meta=json.loads(row[3]);language=str(meta.get('language') or 'unknown').lower()
        if language not in ('en','english','unknown') or not row[2]: return None
        year=meta.get('year') or meta.get('publication_year')
        try: year=int(year) if year else None
        except (TypeError,ValueError): year=None
        if self.from_year and (year is None or year<self.from_year): return None
        if self.to_year and (year is None or year>self.to_year): return None
        return {'doc_id':row[0],'doi':normalize_doi(meta.get('doi') or row[0]),'title':row[1],'abstract':row[2],
                'year':year,'journal':meta.get('journal'),'publication_date':meta.get('publication_date'),
                'authors':meta.get('authors_wos',meta.get('authors')),'language':language,
                'cited_by_count':meta.get('openalex_cited_by_count',meta.get('cited_by_count')),
                'topics':meta.get('openalex_topics',[]),'keywords':meta.get('openalex_keywords',[]),
                'source':'local_corpus','retrieved_at_utc':utc_now()}

    def lexical(self,question,limit):
        words=re.findall(r'[A-Za-z0-9]+',question['search_query'])
        stop={'the','and','or','of','in','on','for','a','an','to','how','what','which','with','are','is','do','does','compare','studies','research','methods','data'}
        words=[w for w in words if w.lower() not in stop][:18]
        fallback=' OR '.join('"'+w+'"' for w in words)
        expression=question.get('fts_expression') or fallback
        if not expression: return [],'Empty English query'
        def ranked(expr,budget):
            if not self.from_year and not self.to_year:
                return self.db.execute('SELECT chunk_id,rank FROM chunk_fts WHERE chunk_fts MATCH ? ORDER BY rank LIMIT ?',(expr,budget)).fetchall()
            clauses=[];params=[expr]
            year="CAST(COALESCE(json_extract(d.metadata_json,'$.year'),json_extract(d.metadata_json,'$.publication_year')) AS INTEGER)"
            if self.from_year: clauses.append(year+'>=?');params.append(self.from_year)
            if self.to_year: clauses.append(year+'<=?');params.append(self.to_year)
            params.append(budget)
            return self.db.execute('SELECT chunk_fts.chunk_id,chunk_fts.rank FROM chunk_fts JOIN documents d ON d.doc_id=chunk_fts.doc_id WHERE chunk_fts MATCH ? AND '+' AND '.join(clauses)+' ORDER BY chunk_fts.rank LIMIT ?',params).fetchall()
        try: hits=ranked(expression,max(120,limit*5));note=None
        except sqlite3.OperationalError:
            hits=ranked(fallback,max(120,limit*5));note='Invalid model-produced FTS expression; documented fallback to token OR query'
        seeds=[];seen=set()
        for chunk,_ in hits:
            row=self.db.execute('SELECT doc_id FROM chunks WHERE chunk_id=?',(int(chunk),)).fetchone()
            if not row or row[0] in seen or not self.paper(row[0]): continue
            seen.add(row[0]);seeds.append({'doc_id':row[0],'score':1/(60+len(seeds)+1),'graph_paths':[]})
            if len(seeds)>=limit: break
        return seeds,note

    def expand(self,seeds,limit):
        if not self.graph_available: return seeds[:limit]
        weights={'author':1.,'institution':.45,'topic':.85,'keyword':.65}
        scores=defaultdict(float);paths=defaultdict(list)
        for seed in seeds[:24]:
            sid=seed['doc_id'];score=seed['score']
            for relation,key,label in self.db.execute('SELECT relation,entity_key,entity_label FROM graph_memberships WHERE doc_id=?',(sid,)):
                count=self.db.execute('SELECT COUNT(*) FROM graph_memberships WHERE relation=? AND entity_key=?',(relation,key)).fetchone()[0]
                cap=1500 if relation in ('topic','keyword') else 500
                if not 2<=count<=cap: continue
                for (peer,) in self.db.execute('SELECT doc_id FROM graph_memberships WHERE relation=? AND entity_key=? AND doc_id!=? LIMIT 300',(relation,key,sid)):
                    scores[peer]+=score*weights.get(relation,.5)/math.log2(count+1)
                    if len(paths[peer])<10: paths[peer].append({'relation':relation,'label':label or key,'seed_doi':sid})
            for sql,relation,weight in (
                ('SELECT target_doc_id FROM citation_edges WHERE source_doc_id=? LIMIT 300','seed_cites_candidate',1.35),
                ('SELECT source_doc_id FROM citation_edges WHERE target_doc_id=? LIMIT 300','candidate_cites_seed',1.15)):
                for (peer,) in self.db.execute(sql,(sid,)):
                    scores[peer]+=score*weight
                    if len(paths[peer])<10: paths[peer].append({'relation':relation,'seed_doi':sid})
        fused={s['doc_id']:dict(s) for s in seeds};highest=max(scores.values(),default=1);base=max((s['score'] for s in seeds),default=1)
        for peer in sorted(scores,key=scores.get,reverse=True)[:limit*3]:
            if not self.paper(peer): continue
            item=fused.setdefault(peer,{'doc_id':peer,'score':0,'graph_paths':[]})
            item['score']+=.8*base*scores[peer]/highest;item['graph_paths']=paths[peer]
        return sorted(fused.values(),key=lambda s:s['score'],reverse=True)[:limit]

    def retrieve(self,question,top_k=4,candidate_k=40):
        seeds,note=self.lexical(question,candidate_k)
        modes={}
        for strategy in ('graph','ordinary'):
            ranked=self.expand(seeds,candidate_k) if strategy=='graph' else seeds
            papers=[]
            for hit in ranked:
                p=self.paper(hit['doc_id'])
                if p and p['doi']:
                    papers.append(p|{'graph_paths':hit['graph_paths']})
                    if len(papers)>=top_k: break
            modes[strategy]=papers
        return {'query_id':question['id'],'question':question['question'],'search_query':question['search_query'],
            'fts_expression':question.get('fts_expression'),'retrieval_mode':'lexical_fts5',
            'shared_seed_dois':[s['doc_id'] for s in seeds],'graph_available':self.graph_available,
            'fallback_note':note,'modes':modes}


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    imp=sub.add_parser('import');imp.add_argument('--source',type=Path,required=True);imp.add_argument('--database',type=Path,required=True)
    get=sub.add_parser('retrieve');get.add_argument('--database',type=Path,required=True);get.add_argument('--plan',type=Path,required=True);get.add_argument('--output',type=Path,required=True);get.add_argument('--top-k',type=int,default=4)
    args=p.parse_args()
    if args.command=='import':
        import_jsonl(args.source,args.database,lambda stage,done,total:print(f'{stage}: {done:,}/{total:,} bytes',flush=True));return
    questions=json.loads(args.plan.read_text(encoding='utf-8'))['questions'];args.output.mkdir(parents=True,exist_ok=True)
    rag=Retriever(args.database);papers={}
    try:
        with (args.output/'retrieval.jsonl').open('w',encoding='utf-8') as f:
            for i,q in enumerate(questions,1):
                event=rag.retrieve(q,args.top_k);f.write(json.dumps(event,ensure_ascii=False)+'\n');f.flush()
                for mode,items in event['modes'].items():
                    for item in items:
                        row=papers.setdefault(item['doi'],item|{'source_query':[]})
                        row['source_query'].append({'query_id':q['id'],'strategy':mode,'graph_paths':item['graph_paths']})
                print(f'Retrieval: {i}/{len(questions)} | {q["id"]}',flush=True)
        (args.output/'documents.jsonl').write_text(''.join(json.dumps(p,ensure_ascii=False)+'\n' for p in papers.values()),encoding='utf-8')
        write_json(args.output/'retrieval_manifest.json',{'created_at_utc':utc_now(),'unique_papers':len(papers),'retrieval_mode':'lexical_fts5','order':['graph','ordinary'],'corpus_scope':rag.scope(),'status':'Candidates requiring source reading'})
    finally: rag.close()

if __name__=='__main__': main()

"""Portable command entry point. No fixed project paths or embedded API keys."""
import argparse
import json
import sys
from pathlib import Path
from .core import ReviewAgent,DEFAULTS


def load_settings(path=None):
    settings=DEFAULTS.copy()
    if path: settings.update(json.loads(Path(path).read_text(encoding='utf-8-sig')))
    return settings


def main():
    p=argparse.ArgumentParser(description='Urban/rural planning literature review agent')
    p.add_argument('--config',type=Path)
    sub=p.add_subparsers(dest='command')
    review=sub.add_parser('review');review.add_argument('--statement-file',required=True,type=Path);review.add_argument('--output',required=True,type=Path)
    recent=sub.add_parser('recent');recent.add_argument('--query',required=True);recent.add_argument('--from-year',type=int,default=2026);recent.add_argument('--limit',type=int,default=20);recent.add_argument('--output',type=Path,required=True)
    imp=sub.add_parser('import');imp.add_argument('--source',type=Path,required=True);imp.add_argument('--database',type=Path,required=True)
    sub.add_parser('mcp')
    retrieve=sub.add_parser('retrieve');retrieve.add_argument('--plan',type=Path,required=True);retrieve.add_argument('--output',type=Path,required=True)
    service=sub.add_parser('serve');service.add_argument('--port',type=int,default=8765)
    chat=sub.add_parser('chat');chat.add_argument('--port',type=int,default=8501);chat.add_argument('--host',default='127.0.0.1')
    sub.add_parser('gui')
    args=p.parse_args();settings=load_settings(args.config)
    if args.command in (None,'gui'):
        from .gui import launch
        launch(settings);return
    if args.command=='review':
        ReviewAgent(settings).run(args.statement_file.read_text(encoding='utf-8-sig'),args.output)
    elif args.command=='recent':
        from .live import OpenAlex
        from .retriever import write_json
        write_json(args.output,OpenAlex(key_env=settings.get('openalex_key_env','OPENALEX_API_KEY')).search(args.query,args.from_year,limit=args.limit))
        print('Discovery records saved. Read titles and abstracts before using them in a review.',flush=True)
    elif args.command=='import':
        from .retriever import import_jsonl
        import_jsonl(args.source,args.database,lambda stage,done,total:print(f'{stage}: {done:,}/{total:,} bytes',flush=True))
    elif args.command=='retrieve':
        from .backends import Backend
        from .retriever import write_json,utc_now
        plan=json.loads(args.plan.read_text(encoding='utf-8-sig'))
        rag=Backend(settings);args.output.mkdir(parents=True,exist_ok=True);papers={};jobs=[]
        try:
            for q in plan['questions']:
                job=rag.retrieve(q,settings['top_k'],settings['candidate_k']);jobs.append(job)
                ReviewAgent._merge_sources(papers,job)
                write_json(args.output/'retrieval_jobs.json',jobs);write_json(args.output/'retrieved_sources.json',papers)
                print('Saved '+q['id']+f'; {len(papers)} unique DOI-linked abstracts',flush=True)
            write_json(args.output/'retrieval_manifest.json',{'created_at_utc':utc_now(),'corpus_scope':rag.scope(),'unique_candidates':len(papers),'status':'Source reading required'})
        finally: rag.close()
    elif args.command=='mcp':
        from .mcp_server import serve_mcp
        serve_mcp(settings)
    elif args.command=='serve':
        from .service import serve
        serve(settings,port=args.port)
    elif args.command=='chat':
        import os
        import subprocess
        import shutil
        app=Path(__file__).with_name('chat_app.py')
        config=Path(args.config).resolve() if args.config else Path('planning-agent.json').resolve()
        env=os.environ.copy();env['PLANNING_REVIEW_CONFIG']=str(config)
        streamlit=shutil.which('streamlit')
        if streamlit:
            command=[streamlit,'run',str(app),'--server.address',args.host,'--server.port',str(args.port)]
            if args.host in ('127.0.0.1','localhost','::1'): command.append('--server.headless=true')
            raise SystemExit(subprocess.call(command,env=env))
        raise RuntimeError('Streamlit is missing. Install distribution/requirements-chat.txt and retry.')


if __name__=='__main__':
    try: main()
    except Exception as exc:
        print(str(exc),file=sys.stderr);sys.exit(1)

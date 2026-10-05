"""Prepare the lossless bundled corpus with bounded memory and atomic activation."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time
from .retriever import write_json,utc_now


def application_root():
    return Path(sys.executable).resolve().parent if getattr(sys,'frozen',False) else Path(__file__).resolve().parents[1]


def discover_bundle():
    base=application_root()
    for folder in (base/'corpus',base/'corpus_payload_0.2.0'):
        path=folder/'bundle_manifest.json'
        if path.is_file():
            value=json.loads(path.read_text(encoding='utf-8'))
            if value.get('status')=='complete': return folder,value
    return None,None


def default_library(version='0.2.0'):
    return Path(os.environ.get('LOCALAPPDATA',str(Path.home())))/'PlanningReviewAgent'/'corpus'/version


def prepare_bundle(folder,target,include_vectors=False,cancel=None,emit=None):
    import zstandard as zstd
    from .core import RunLock
    folder=Path(folder).resolve();target=Path(target).resolve()
    cancel=cancel or threading.Event();emit=emit or (lambda e:None)
    manifest=json.loads((folder/'bundle_manifest.json').read_text(encoding='utf-8'))
    if manifest.get('status')!='complete': raise ValueError('The bundled corpus is incomplete.')
    chosen=[f for f in manifest['files'] if include_vectors or not f.get('optional')]
    if any(f['output_name'] not in ('rag.sqlite','embeddings.f32') or Path(f['filename']).name!=f['filename'] for f in chosen):
        raise ValueError('Invalid bundle filenames.')
    target.mkdir(parents=True,exist_ok=True)
    receipt_path=target/'prepared_corpus.json';started=time.monotonic();last=[0.]
    with RunLock(target/'.prepare.lock'):
        receipt=json.loads(receipt_path.read_text(encoding='utf-8')) if receipt_path.exists() else {'files':{}}
        prepared=receipt.get('files',{})
        pending=[]
        for item in chosen:
            output=target/item['output_name']
            prior=prepared.get(item['output_name'],{})
            if output.exists():
                if prior.get('sha256')==item['uncompressed_sha256'] and output.stat().st_size==item['uncompressed_bytes']: continue
                raise ValueError('An existing file is not a verified copy of this bundle. Choose a new library folder; existing files are never overwritten.')
            pending.append(item)
        total=sum(x['uncompressed_bytes'] for x in pending)
        if shutil.disk_usage(target).free<total+256*1024*1024:
            raise ValueError(f'Insufficient free disk space. This selection needs about {total/1e9:.1f} GB plus working space.')
        done=0
        def report(message,force=False):
            if force or time.monotonic()-last[0]>=3:
                elapsed=time.monotonic()-started
                eta=elapsed/done*(total-done) if done else None
                emit({'kind':'progress','status':'preparing_corpus','message':message,
                    'completed_steps':done,'total_steps':max(total,1),'eta_seconds':eta,'unit':'bytes'})
                last[0]=time.monotonic()
        for item in pending:
            if cancel.is_set(): raise InterruptedError('Library preparation stopped. Completed files remain verified; the interrupted file restarts next time.')
            source=folder/item['filename'];output=target/item['output_name'];partial=output.with_name(output.name+'.unpacking')
            if source.stat().st_size!=item['compressed_bytes']: raise ValueError('A compressed bundle file is incomplete. Obtain the complete package.')
            digest=hashlib.sha256();written=0;report('Preparing '+item['output_name'],True)
            with source.open('rb') as raw,partial.open('wb') as dst:
                with zstd.ZstdDecompressor().stream_reader(raw,closefd=False) as reader:
                    while chunk:=reader.read(4*1024*1024):
                        if cancel.is_set(): raise InterruptedError('Library preparation stopped. Run Prepare again to restart the interrupted file.')
                        written+=len(chunk)
                        if written>item['uncompressed_bytes']: raise ValueError('Expanded file exceeds its manifest size.')
                        digest.update(chunk);dst.write(chunk);done+=len(chunk);report('Preparing '+item['output_name'])
                dst.flush();os.fsync(dst.fileno())
            if written!=item['uncompressed_bytes'] or digest.hexdigest()!=item['uncompressed_sha256']:
                raise ValueError('Expanded file failed its size/SHA-256 integrity check. It has not been activated.')
            partial.replace(output)
            prepared[item['output_name']]={'sha256':digest.hexdigest(),'bytes':written,'verified_at_utc':utc_now()}
            write_json(receipt_path,{'bundle_version':manifest['version'],'files':prepared})
            report('Verified '+item['output_name'],True)
        for name in manifest.get('sidecars',[]):
            if Path(name).name!=name or not name.endswith('.json'): raise ValueError('Invalid sidecar name.')
            shutil.copy2(folder/name,target/name)
        report('Library ready',True)
    return target/'rag.sqlite'

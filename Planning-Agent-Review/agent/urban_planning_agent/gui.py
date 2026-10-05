"""Planning Studio: styled review workspace with a losslessly bundled library."""
from __future__ import annotations
import json
import os
import queue
import threading
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog,messagebox,ttk
from .core import DEFAULTS,ReviewAgent,Ollama,fmt_time
from .retriever import import_jsonl,write_json
from .bundle import discover_bundle,default_library,prepare_bundle

BG='#F3F6FA';WHITE='#FFFFFF';INK='#162C43';MUTED='#63758B';TEAL='#087F8C';NAVY='#122B43'

def launch(initial=None):
    root=tk.Tk();root.title('Planning Studio | Literature Review Agent')
    width=min(1200,max(1030,root.winfo_screenwidth()-100));height=min(860,max(730,root.winfo_screenheight()-100))
    root.geometry(f'{width}x{height}');root.minsize(1030,730);root.configure(bg=BG)
    style=ttk.Style(root);style.theme_use('clam')
    style.configure('.',font=('Segoe UI',10),foreground=INK)
    style.configure('TFrame',background=WHITE);style.configure('TLabel',background=WHITE)
    style.configure('TButton',padding=(13,9),background='#EAF0F5',borderwidth=0)
    style.map('TButton',background=[('active','#DCE7EE')])
    style.configure('Primary.TButton',background=TEAL,foreground=WHITE,font=('Segoe UI',10,'bold'),padding=(17,10))
    style.map('Primary.TButton',background=[('active','#086D79')])
    style.configure('Nav.TButton',background=NAVY,foreground='#BDD1DF',padding=(14,14),anchor='w')
    style.map('Nav.TButton',background=[('active','#204A62')],foreground=[('active',WHITE)])
    style.configure('Selected.Nav.TButton',background='#204A62',foreground=WHITE,padding=(14,14),anchor='w')
    style.configure('TEntry',padding=8,fieldbackground=WHITE)
    style.configure('TCombobox',padding=7,fieldbackground=WHITE)
    style.configure('TCheckbutton',background=WHITE,padding=4)
    style.configure('Studio.TNotebook',background=BG,borderwidth=0)
    style.layout('Studio.TNotebook.Tab',[])
    style.configure('Studio.Horizontal.TProgressbar',background=TEAL,troughcolor='#E7EEF4',borderwidth=0,thickness=10)
    private=Path(os.environ.get('APPDATA',str(Path.home())))/'PlanningReviewAgent'/'settings.json'
    settings=DEFAULTS.copy()
    if private.exists():
        try: settings.update(json.loads(private.read_text(encoding='utf-8')))
        except (ValueError,OSError): pass
    if initial: settings.update({k:v for k,v in initial.items() if k not in DEFAULTS or v!=DEFAULTS[k]})
    try: bundle,manifest=discover_bundle()
    except (ValueError,OSError): bundle,manifest=None,None
    events=queue.Queue();state={'worker':None,'cancel':threading.Event(),'output':None}
    variables={k:tk.StringVar(value=str(settings.get(k) or '')) for k in ('backend','database','service_url','ollama_url','model','output_root')}
    if settings.get('output_root')=='reviews': variables['output_root'].set(str(Path.home()/'PlanningReviews'))
    library_dir=tk.StringVar(value=str(default_library((manifest or {}).get('version','0.2.0'))))
    if not variables['database'].get() and (Path(library_dir.get())/'prepared_corpus.json').is_file():
        if (Path(library_dir.get())/'rag.sqlite').is_file(): variables['database'].set(str(Path(library_dir.get())/'rag.sqlite'))
    live=tk.BooleanVar(value=bool(settings.get('live_enabled')));vectors=tk.BooleanVar(value=False)
    sidebar=tk.Frame(root,bg=NAVY,width=232);sidebar.pack(side='left',fill='y');sidebar.pack_propagate(False)
    tk.Label(sidebar,text='PLANNING',bg=NAVY,fg=WHITE,font=('Segoe UI',21,'bold')).pack(anchor='w',padx=25,pady=(34,0))
    tk.Label(sidebar,text='RESEARCH STUDIO',bg=NAVY,fg='#7FC9CB',font=('Segoe UI',10,'bold')).pack(anchor='w',padx=26,pady=(3,28))
    body=tk.Frame(root,bg=BG);body.pack(side='right',fill='both',expand=True,padx=25,pady=24)
    tk.Label(body,text='Evidence to insight.',bg=BG,fg=INK,font=('Segoe UI',25,'bold')).pack(anchor='w')
    tk.Label(body,text='Planning literature, grounded in traceable sources.',bg=BG,fg=MUTED,font=('Segoe UI',11)).pack(anchor='w',pady=(5,20))
    book=ttk.Notebook(body,style='Studio.TNotebook');book.pack(fill='both',expand=True)
    pages=[ttk.Frame(book,padding=22) for _ in range(3)]
    for page,name in zip(pages,['Review','Library','Connections']): book.add(page,text=name)
    nav_buttons=[]
    def navigate(index):
        book.select(index)
        for i,b in enumerate(nav_buttons): b.configure(style='Selected.Nav.TButton' if i==index else 'Nav.TButton')
    for i,label in enumerate(['01   Literature review','02   Your library','03   Connections']):
        b=ttk.Button(sidebar,text=label,style='Nav.TButton',command=lambda index=i:navigate(index));b.pack(fill='x',padx=12,pady=4);nav_buttons.append(b)
    tk.Frame(sidebar,bg='#2B455B',height=1).pack(fill='x',padx=25,pady=24)
    for title,text in [('SCOPE','Regions · research objects\nOpen-ended methods'),('EVIDENCE','DOI-linked abstracts\nGraph provenance'),('SYNTHESIS','English review drafts\nComparisons and limitations')]:
        tk.Label(sidebar,text=title,bg=NAVY,fg='#7FC9CB',font=('Segoe UI',9,'bold')).pack(anchor='w',padx=26,pady=(9,5))
        tk.Label(sidebar,text=text,bg=NAVY,fg='#BCD0DF',justify='left').pack(anchor='w',padx=26,pady=(0,10))
    tk.Label(sidebar,text='PERSONAL AGENT  /  v0.2.0',bg=NAVY,fg='#7894A8',font=('Segoe UI',8)).pack(side='bottom',anchor='w',padx=24,pady=28)
    def heading(page,title,text):
        ttk.Label(page,text=title,font=('Segoe UI',17,'bold')).pack(anchor='w')
        ttk.Label(page,text=text,foreground=MUTED,wraplength=790).pack(anchor='w',pady=(6,18))
    review,library,connections=pages
    heading(review,'Build your literature review','Describe the planning question, study regions and evidence you want to compare.')
    ttk.Label(review,text='RESEARCH STATEMENT',font=('Segoe UI',9,'bold'),foreground=MUTED).pack(anchor='w',pady=(0,6))
    statement=tk.Text(review,height=5,wrap='word',font=('Segoe UI',11),bg='#F8FAFC',fg=INK,insertbackground=TEAL,relief='flat',highlightthickness=1,highlightbackground='#DFE7EF',padx=13,pady=11);statement.pack(fill='x')
    ttk.Label(review,text='Input: Chinese or English. Saved research plans, evidence and reviews: English.',foreground=MUTED,font=('Segoe UI',9)).pack(anchor='w',pady=(8,12))
    ttk.Checkbutton(review,text='Supplement with 2026+ OpenAlex literature',variable=live).pack(anchor='w')
    ttk.Label(review,text='English Article/Review · DOI · at least one citation',foreground=MUTED,font=('Segoe UI',9)).pack(anchor='w',pady=(2,12))
    buttons=ttk.Frame(review);buttons.pack(fill='x',pady=(0,18))
    status=tk.StringVar(value='Ready. Prepare your library and connect a model to begin.')
    ttk.Label(review,textvariable=status,font=('Segoe UI',10,'bold'),wraplength=790).pack(anchor='w')
    bar=ttk.Progressbar(review,style='Studio.Horizontal.TProgressbar',maximum=100);bar.pack(fill='x',pady=(12,18))
    ttk.Label(review,text='ACTIVITY',font=('Segoe UI',9,'bold'),foreground=MUTED).pack(anchor='w',pady=(0,8))
    log_frame=ttk.Frame(review);log_frame.pack(fill='both',expand=True)
    log=tk.Text(log_frame,height=7,wrap='word',state='disabled',font=('Consolas',9),bg='#F5F8FB',fg='#344B62',relief='flat',padx=11,pady=10)
    scroll=ttk.Scrollbar(log_frame,command=log.yview);log.configure(yscrollcommand=scroll.set);scroll.pack(side='right',fill='y');log.pack(side='left',fill='both',expand=True)
    ttk.Label(review,text='Abstract-based drafts require source review before publication.',foreground=MUTED,font=('Segoe UI',9)).pack(anchor='w',pady=(12,0))
    heading(library,'Your planning library','Prepare the bundled corpus once, or connect an existing local index.')
    stats=ttk.Frame(library);stats.pack(fill='x',pady=(0,22))
    values=[('INDEXED PAPERS',f"{manifest['source_documents']:,}" if manifest else 'Bring your corpus'),('ELIGIBLE PAPERS',f"{manifest['eligible_documents']:,}" if manifest else 'Index policy'),('COMPRESSED DATA',f"{manifest['compressed_bytes']/1e9:.2f} GB" if manifest else 'No bundle found')]
    for label,value in values:
        cell=tk.Frame(stats,bg='#F2F7FA',padx=13,pady=15);cell.pack(side='left',fill='x',expand=True,padx=(0,8))
        tk.Label(cell,text=label,bg='#F2F7FA',fg=MUTED,font=('Segoe UI',8,'bold')).pack(anchor='w')
        tk.Label(cell,text=value,bg='#F2F7FA',fg=INK,font=('Segoe UI',16,'bold')).pack(anchor='w',pady=(7,0))
    ttk.Label(library,text='LOCAL LIBRARY FOLDER',font=('Segoe UI',9,'bold'),foreground=MUTED).pack(anchor='w')
    ttk.Entry(library,textvariable=library_dir).pack(fill='x',pady=(8,13))
    ttk.Checkbutton(library,text='Also unpack vectors for the original dense retrieval pipeline (+3.51 GB)',variable=vectors).pack(anchor='w')
    ttk.Label(library,text='The portable GUI uses lexical FTS5 and bibliographic graph expansion. Preserved vectors are available for the original dense pipeline.',foreground=MUTED,wraplength=790).pack(anchor='w',pady=(9,17))
    library_actions=ttk.Frame(library);library_actions.pack(fill='x',pady=(0,18))
    ttk.Label(library,textvariable=status,wraplength=790).pack(anchor='w')
    library_bar=ttk.Progressbar(library,style='Studio.Horizontal.TProgressbar',maximum=100);library_bar.pack(fill='x',pady=(12,18))
    ttk.Label(library,text='Preparation checks file size and SHA-256. Allow approximately 13.6 GB for the database, or 17.1 GB including vectors. Completed files are retained if you stop; an interrupted file restarts.',foreground=MUTED,wraplength=790).pack(anchor='w')
    heading(connections,'Connections & preferences','Choose your evidence source, local model and review output folder.')
    form=ttk.Frame(connections);form.pack(fill='x');form.columnconfigure(1,weight=1);model_picker=None
    for row,(key,label) in enumerate([('backend','Evidence source'),('database','SQLite index'),('service_url','Shared HTTPS service'),('ollama_url','Ollama address'),('model','Generation model'),('output_root','Review output folder')]):
        ttk.Label(form,text=label).grid(row=row,column=0,sticky='w',padx=(0,17),pady=8)
        if key=='backend': widget=ttk.Combobox(form,textvariable=variables[key],values=['local','remote','live'],state='readonly')
        elif key=='model': model_picker=ttk.Combobox(form,textvariable=variables[key]);widget=model_picker
        else: widget=ttk.Entry(form,textvariable=variables[key])
        widget.grid(row=row,column=1,sticky='ew',pady=8)
        if key=='database': ttk.Button(form,text='Browse',command=lambda:browse_file()).grid(row=row,column=2,padx=(8,0))
        if key=='output_root': ttk.Button(form,text='Choose',command=lambda:browse_output()).grid(row=row,column=2,padx=(8,0))
    connection_actions=ttk.Frame(connections);connection_actions.pack(fill='x',pady=20)
    ttk.Label(connections,text='Local: a prepared SQLite corpus. Remote: an owner-operated HTTPS gateway. Live: OpenAlex discovery without a corpus download.',foreground=MUTED,wraplength=790).pack(anchor='w',pady=(8,16))
    ttk.Label(connections,text='Credentials: OPENALEX_API_KEY and PLANNING_SERVICE_TOKEN environment variables. Restart this application after changing them.',foreground=MUTED,wraplength=790).pack(anchor='w')
    navigate(0)
    def append(text):
        log.configure(state='normal');log.insert('end',datetime.now().strftime('%H:%M:%S')+' | '+text+'\n');log.see('end');log.configure(state='disabled')
    def get_settings(): return settings|{k:v.get().strip() for k,v in variables.items()}|{'live_enabled':live.get()}
    def busy(): return state['worker'] is not None and state['worker'].is_alive()
    def background(fn):
        if busy(): messagebox.showinfo('Active job','Wait for this job or stop it before starting another.');return
        state['cancel']=threading.Event()
        def target():
            try: fn();events.put({'kind':'finished'})
            except InterruptedError as exc: events.put({'kind':'finished','message':str(exc)})
            except Exception as exc: events.put({'kind':'error','message':str(exc)})
        state['worker']=threading.Thread(target=target,daemon=True);state['worker'].start()
    def connect():
        cfg=get_settings()
        def task():
            names=Ollama(cfg,Path(cfg['output_root']),state['cancel'],events.put).tags()
            events.put({'kind':'models','names':names,'message':f'Connected: {len(names)} installed model(s). Select an exact model name.'})
        background(task)
    def save():
        write_json(private,get_settings());append('Private local settings saved. No API key or service token is stored in this file.')
    def load_config():
        path=filedialog.askopenfilename(title='Load an owner or client configuration',filetypes=[('JSON configuration','*.json')])
        if not path: return
        try:
            cfg=json.loads(Path(path).read_text(encoding='utf-8-sig'));settings.update(cfg)
            for key,var in variables.items(): var.set(str(cfg.get(key,settings.get(key)) or ''))
            live.set(bool(settings.get('live_enabled')));append('Configuration loaded.')
        except Exception as exc: messagebox.showerror('Configuration error',str(exc))
    def run(resume=False):
        if busy(): messagebox.showinfo('Active task','Wait for the active task or stop it first.');return
        text=statement.get('1.0','end').strip();cfg=get_settings()
        if not text: messagebox.showinfo('Review statement','Enter the topic, regions and desired evidence scope.');return
        if not cfg['model']: messagebox.showinfo('Model selection','Connect to Ollama and select an installed model.');return
        if resume:
            chosen=filedialog.askdirectory(title='Choose the existing review run folder')
            if not chosen: return
            output=Path(chosen)
        else: output=Path(cfg['output_root'])/('review_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
        state['output']=output
        cache_folder=library_dir.get().strip()
        def task():
            if cfg.get('backend')=='local' and not cfg.get('database') and bundle:
                cfg['database']=str(prepare_bundle(bundle,cache_folder,False,state['cancel'],events.put))
                events.put({'kind':'imported','path':cfg['database'],'message':'Bundled corpus prepared for this review.'})
            ReviewAgent(cfg,events.put,state['cancel']).run(text,output)
            events.put({'kind':'result','message':'Complete draft saved: '+str(output/'literature_review_draft.md')})
        background(task)
    def stop():
        state['cancel'].set();append('Stop requested. Completed steps remain saved for resume; a network read may take up to its timeout.')
    def open_output():
        path=state['output'] or Path(variables['output_root'].get())
        if path.exists(): os.startfile(str(path))
    def import_corpus():
        source=filedialog.askopenfilename(title='Import English DOI-linked metadata',filetypes=[('JSON Lines','*.jsonl')])
        if not source: return
        target=filedialog.asksaveasfilename(title='Choose a NEW index file',defaultextension='.sqlite',filetypes=[('SQLite','*.sqlite')])
        if not target: return
        def progress(stage,done,total): events.put({'kind':'progress','completed_steps':done,'total_steps':total,'eta_seconds':None,'unit':'bytes','message':f'Import {stage}: {done:,}/{total:,} bytes'})
        def task():
            import_jsonl(source,target,progress,state['cancel']);events.put({'kind':'imported','path':target,'message':'New corpus index imported.'})
        background(task)
    def choose_library():
        path=filedialog.askdirectory(title='Choose a local library folder')
        if path: library_dir.set(path)
    def prepare_library():
        if not bundle:
            messagebox.showinfo('Corpus bundle','This software-only package has no compressed corpus. Choose an existing index or use the corpus-inclusive package.');return
        target=library_dir.get().strip();include=vectors.get()
        def task():
            database=prepare_bundle(bundle,target,include,state['cancel'],events.put)
            events.put({'kind':'imported','path':str(database),'message':'Verified corpus ready for local retrieval.'})
        background(task)
    for label,callback,primary in [('Start review',run,True),('Resume',lambda:run(True),False),('Stop',stop,False),('Open output',open_output,False)]:
        ttk.Button(buttons,text=label,command=callback,style='Primary.TButton' if primary else 'TButton').pack(side='left',padx=(0,8))
    for label,callback in [('Connect / Models',connect),('Load config',load_config),('Save settings',save)]:
        ttk.Button(connection_actions,text=label,command=callback).pack(side='left',padx=(0,8))
    ttk.Button(library_actions,text='Prepare bundled corpus',style='Primary.TButton',command=prepare_library).pack(side='left',padx=(0,8))
    ttk.Button(library_actions,text='Choose folder',command=choose_library).pack(side='left',padx=(0,8))
    ttk.Button(library_actions,text='Import JSONL',command=import_corpus).pack(side='left')
    ttk.Button(library_actions,text='Stop',command=stop).pack(side='left',padx=(8,0))
    def poll():
        try:
            while True:
                event=events.get_nowait();kind=event.get('kind');message=event.get('message','')
                if message: append(message)
                if kind=='models':
                    model_picker['values']=event['names']
                    if event['names'] and not variables['model'].get(): variables['model'].set(event['names'][0])
                if kind=='imported': variables['database'].set(event['path']);variables['backend'].set('local');status.set(message)
                if kind=='progress':
                    total=event.get('total_steps') or 1;done=event.get('completed_steps',0);bar['value']=min(100,done/total*100);library_bar['value']=bar['value']
                    amount=f'{done/1e9:.2f}/{total/1e9:.2f} GB' if event.get('unit')=='bytes' else f'{done:,}/{total:,} stages'
                    status.set(f'{message} | {done/total*100:.1f}% | {amount} | ETA {fmt_time(event.get("eta_seconds"))}')
                elif kind in ('finished','error','result') and message: status.set(message)
        except queue.Empty: pass
        root.after(250,poll)
    def close(): state['cancel'].set();root.destroy()
    if manifest: append(f"Bundled library detected: {manifest['eligible_documents']:,} eligible papers. Prepare it in Your library.")
    root.protocol('WM_DELETE_WINDOW',close);root.after(250,poll);root.mainloop()

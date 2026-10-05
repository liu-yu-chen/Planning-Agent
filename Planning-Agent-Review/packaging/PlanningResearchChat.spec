# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
from PyInstaller.utils.hooks import collect_all

HERE = Path(SPECPATH)
datas = [(str(HERE / 'urban_planning_agent'), 'distribution/urban_planning_agent')]
binaries = []
hiddenimports = [
    'urban_planning_agent',
    'urban_planning_agent.core',
    'urban_planning_agent.retriever',
    'urban_planning_agent.backends',
    'urban_planning_agent.live',
    'urban_planning_agent.bundle',
]
for package in ('streamlit', 'llama_index.core', 'llama_index.llms.ollama', 'ollama', 'zstandard'):
    package_datas, package_binaries, package_hidden = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hidden

hiddenimports += ['PySide6.QtCore', 'PySide6.QtGui', 'PySide6.QtWidgets']

a = Analysis(
    [str(HERE / 'chat_launcher.py')],
    pathex=[str(HERE)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['mcp'],
    noarchive=False,
    optimize=1,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='PlanningResearchChat',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='PlanningResearchChat',
)

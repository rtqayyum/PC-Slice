"""Export manuscript TikZ drawings as separate vector PDFs without running experiments."""
from pathlib import Path
import shutil
import subprocess

root=Path(__file__).resolve().parent
preamble=r'''\documentclass[tikz,border=3pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb,tikz,pgfplots,pgfplotstable}
\usetikzlibrary{arrows.meta,positioning,fit,backgrounds,calc}
\pgfplotsset{compat=1.18,every axis/.append style={font=\footnotesize,label style={font=\footnotesize},tick label style={font=\scriptsize},legend style={font=\scriptsize},grid=major,grid style={gray!15}}}
\definecolor{pcblue}{RGB}{20,77,124}
\definecolor{pcorange}{RGB}{196,95,20}
\definecolor{pcgreen}{RGB}{28,116,88}
\definecolor{pcgray}{RGB}{95,101,110}
\input{plot_style.tex}
\setlength{\textwidth}{17.25cm}
\setlength{\columnwidth}{8.415cm}
\begin{document}
'''
engine=shutil.which('tectonic')
if not engine:raise SystemExit('Install Tectonic to export the individual figures.')
(root/'pdf').mkdir(exist_ok=True)
(root/'build/figure_sources').mkdir(exist_ok=True)
(root/'build/figure_sources/data').mkdir(exist_ok=True)
for datafile in (root/'data').glob('*.dat'):
    shutil.copy2(datafile,root/'build/figure_sources/data'/datafile.name)
shutil.copy2(root/'plot_style.tex',root/'build/figure_sources/plot_style.tex')
for name in ['architecture','system','completion','network','distributions','failure','computation','policy','rag_latency','activation']:
    src=(root/f'{name}.tex').read_text()
    picture=src[src.index(r'\begin{tikzpicture}'):src.index(r'\end{tikzpicture}')+len(r'\end{tikzpicture}')]
    path=root/'build/figure_sources'/f'{name}.tex'
    path.write_text(preamble+picture+'\n'+r'\end{document}'+'\n')
    subprocess.run([engine,'--outdir','pdf',str(path.relative_to(root))],cwd=root,check=True)

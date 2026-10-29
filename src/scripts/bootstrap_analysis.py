"""
Bootstrap confidence intervals and paired RAG vs. full-context comparisons
computed from the stored per-question scores.

For each model x category x metric: 95% percentile-bootstrap CI over
questions for each condition, and a paired CI on (RAG - full-context)
using the same resampled questions. Binary also gets pooled macro-F1
(same definition as evaluate_results.py), accuracy, and an exact two-sided
McNemar test. Continuous-metric p-values come from the bootstrap and are
approximate. Single deterministic run per configuration, so the CIs only
reflect question sampling; no multiple-comparison correction is applied.

Reads:  results/metrics/per_question_scores.csv
Writes: results/metrics/bootstrap_results.csv

Usage (from anywhere):
    python src/scripts/bootstrap_analysis.py
"""
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
OUT_DIR = ROOT / "results" / "metrics"

P=pd.read_csv('results/metrics/per_question_scores.csv')
print("errors:",P.error.notna().sum(),"| dup (model,strategy,codigo,question):",P.duplicated(['model_key','context_strategy','codigo','question']).sum())
print(P.context_strategy.unique())
B=10000;rng=np.random.default_rng(42)
def lab(x): return {'si':0,'no':1}.get(x,-1) if isinstance(x,str) else -1
def macro_f1(g,p):
    f=[]
    for c in (0,1):
        tp=((p==c)&(g==c)).sum();fp=((p==c)&(g!=c)).sum();fn=((g==c)&(p!=c)).sum()
        pr=tp/(tp+fp) if tp+fp else 0;rc=tp/(tp+fn) if tp+fn else 0
        f.append(2*pr*rc/(pr+rc) if pr+rc else 0)
    return np.mean(f)
def mcnemar(a,b):  # a,b correct arrays; exact two-sided
    n10=int(((a==1)&(b==0)).sum());n01=int(((a==0)&(b==1)).sum());n=n10+n01
    if n==0:return 1.0,n10,n01
    k=min(n10,n01);p=sum(math.comb(n,i) for i in range(k+1))/2**n*2
    return min(1.0,p),n10,n01
rows=[]
for mk in sorted(P.model_key.unique()):
  for cat,metrics in (('binary',['macro_f1','accuracy']),('short_answer',['token_f1','bertscore_f1']),('open_ended',['bertscore_f1','meteor'])):
    d={}
    for s in ('rag','fullcontext'):
        g=P[(P.model_key==mk)&(P.category==cat)&(P.context_strategy==s)].drop_duplicates(['codigo','question']).set_index(['codigo','question']).sort_index()
        d[s]=g
    idx=d['rag'].index.intersection(d['fullcontext'].index)
    a,b=d['rag'].loc[idx],d['fullcontext'].loc[idx];n=len(idx)
    ix=rng.integers(0,n,(B,n))
    for m in metrics:
        if cat=='binary':
            ga=a.expected_binary_label.map(lab).values;pa=a.predicted_binary_label.map(lab).values;pb=b.predicted_binary_label.map(lab).values;gb=b.expected_binary_label.map(lab).values
            if m=='macro_f1':
                f=lambda g,p,i:macro_f1(g[i],p[i])
                va=np.array([f(ga,pa,i) for i in ix]);vb=np.array([f(gb,pb,i) for i in ix]);ma,mb=macro_f1(ga,pa),macro_f1(gb,pb)
            else:
                ca=(pa==ga).astype(float);cb=(pb==gb).astype(float)
                va=ca[ix].mean(1);vb=cb[ix].mean(1);ma,mb=ca.mean(),cb.mean()
            pm=mcnemar((pa==ga).astype(int),(pb==gb).astype(int))
        else:
            xa=a[m].astype(float).values;xb=b[m].astype(float).values
            va=xa[ix].mean(1);vb=xb[ix].mean(1);ma,mb=xa.mean(),xb.mean();pm=(np.nan,0,0)
        dd=va-vb
        lo,hi=np.percentile(dd,[2.5,97.5]);pb2=min(1,2*min((dd<=0).mean(),(dd>=0).mean()))
        rows.append(dict(model=mk,cat=cat,metric=m,n=n,rag=ma,rag_lo=np.percentile(va,2.5),rag_hi=np.percentile(va,97.5),fc=mb,fc_lo=np.percentile(vb,2.5),fc_hi=np.percentile(vb,97.5),diff=ma-mb,d_lo=lo,d_hi=hi,p_boot=pb2,p_mcnemar=pm[0],rag_only_correct=pm[1],fc_only_correct=pm[2]))
R=pd.DataFrame(rows);R['sig']=(R.d_lo>0)|(R.d_hi<0)
R.to_csv(OUT_DIR / 'bootstrap_results.csv',index=False)
pd.set_option('display.width',250)
f=R.copy()
f['RAG']=f.apply(lambda r:f"{r.rag:.3f} [{r.rag_lo:.3f},{r.rag_hi:.3f}]",axis=1)
f['FullCtx']=f.apply(lambda r:f"{r.fc:.3f} [{r.fc_lo:.3f},{r.fc_hi:.3f}]",axis=1)
f['RAG-FC']=f.apply(lambda r:f"{r['diff']:+.3f} [{r.d_lo:+.3f},{r.d_hi:+.3f}]",axis=1)
f['p_mcn']=f.p_mcnemar.map(lambda x:'' if pd.isna(x) else f"{x:.4f}")
print(f[['model','cat','metric','n','RAG','FullCtx','RAG-FC','sig','p_boot','p_mcn']].to_string(index=False,float_format=lambda x:f"{x:.4f}"))

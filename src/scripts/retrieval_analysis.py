"""
Standalone retrieval evaluation (document-level Hit@k and MRR) computed
from the retrieved_sources logged in results/raw/, plus RAG answer quality
split by whether the correct course document was retrieved.

A hit means one of the top-5 retrieved chunks comes from the document of
the course the question was written for (file name starts with the
question's 'codigo'). Retrieval is model-independent, so it is measured
once. No model is run; this only reads stored outputs.

Reads:  results/raw/*.csv, results/metrics/per_question_scores.csv
Writes: results/metrics/retrieval_per_question.csv

Usage (from anywhere):
    python src/scripts/retrieval_analysis.py
"""
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
OUT_DIR = ROOT / "results" / "metrics"

dfs={os.path.basename(f)[:-4]:pd.read_csv(f) for f in glob.glob('results/raw/*.csv')}
ref=dfs['qwen2.5-3b']
# retrieval identical across models?
key=lambda d:(d['codigo']+'|'+d['question']+'|'+d['curso'])
same=all((key(d).values==key(ref).values).all() and (d['retrieved_sources'].values==ref['retrieved_sources'].values).all() for d in dfs.values())
print("retrieval identical across models/rows:",same, {k:len(v) for k,v in dfs.items()})
rows=[]
for i,r in ref.iterrows():
    src=str(r['retrieved_sources']).split(';')
    rank=next((j+1 for j,s in enumerate(src) if s.startswith(r['codigo']+'-')),None)
    rows.append(dict(category=r['category'],codigo=r['codigo'],question=r['question'],rank=rank,n_src=len(src),n_unique=len(set(src))))
R=pd.DataFrame(rows)
R['hit5']=R['rank'].notna(); R['hit1']=R['rank']==1; R['hit3']=R['rank'].le(3)
R['rr']=R['rank'].map(lambda x:0 if pd.isna(x) else 1/x)
def summ(g): return pd.Series(dict(n=len(g),hit1=g.hit1.mean(),hit3=g.hit3.mean(),hit5=g.hit5.mean(),mrr=g.rr.mean()))
print("n_src distribution",R.n_src.value_counts().to_dict(),"unique",R.n_unique.value_counts().to_dict())
print("\nOVERALL");print(summ(R).round(3).to_string())
print("\nBY CATEGORY");print(R.groupby('category').apply(summ).round(3).to_string())
print("\nBY COURSE");print(R.groupby('codigo').apply(summ).round(3).to_string())
# bootstrap CI for overall hit5 / mrr
rng=np.random.default_rng(0);n=len(R);h=R.hit5.values.astype(float);m=R.rr.values
ix=rng.integers(0,n,(10000,n));
print("\nHit@5 95%CI",np.percentile(h[ix].mean(1),[2.5,97.5]).round(3),"MRR 95%CI",np.percentile(m[ix].mean(1),[2.5,97.5]).round(3))
# join with generation correctness (RAG)
P=pd.read_csv('results/metrics/per_question_scores.csv'); P=P[P.context_strategy=='rag'].copy()
P=P.merge(R[['codigo','question','hit5','hit1','rank']].drop_duplicates(['codigo','question']),on=['codigo','question'],how='left')
print("\nmerge nulls",P.hit5.isna().sum(),"dup keys",R.duplicated(['codigo','question']).sum())
P['binary_correct']=P.binary_correct.map({True:1.0,False:0.0,'True':1.0,'False':0.0})
out=[]
for (mk,cat),g in P.groupby(['model_key','category']):
    metric='binary_correct' if cat=='binary' else ('token_f1' if cat=='short_answer' else 'bertscore_f1')
    a=g[g.hit5==True][metric].mean();b=g[g.hit5==False][metric].mean()
    out.append((mk,cat,metric,int((g.hit5==True).sum()),round(a,3),int((g.hit5==False).sum()),round(b,3) if not np.isnan(b) else None))
print("\nRAG quality when correct course IS vs IS NOT in retrieved set");print(pd.DataFrame(out,columns=['model','cat','metric','n_hit','mean_hit','n_miss','mean_miss']).to_string(index=False))
R.to_csv(OUT_DIR / 'retrieval_per_question.csv',index=False)

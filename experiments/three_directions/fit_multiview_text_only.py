"""No-image text-only shortcut baseline for frozen two-view event labels."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.sparse import csr_matrix


DIM = 8192
L2 = .001


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bucket(token: str) -> int:
    return int.from_bytes(hashlib.blake2b(token.encode(),digest_size=8).digest(),
                          "little") % DIM


def vector(instruction: str, clause: str) -> dict[int, float]:
    out=defaultdict(float)
    for prefix,text,weight in (("i",instruction,1.),("g",clause,2.)):
        words=re.findall(r"[a-z0-9]+",text.lower())
        if not words:raise ValueError("empty event instruction")
        for word in words:out[bucket(prefix+":w:"+word)]+=weight/len(words)
        for left,right in zip(words,words[1:]):
            out[bucket(prefix+":b:"+left+"_"+right)]+=weight/max(1,len(words)-1)
    return out


def load(part: str, capture: dict, labels: dict) -> list[dict]:
    if part not in ("fit","development"):raise ValueError("audit unopened")
    plans={x["record_id"]:x for x in capture["selected"][part]}
    result=[]
    for label in labels["selected"][part]:
        p=plans[label["record_id"]]
        wrong=label["kind"]=="wrong_instruction"
        instruction=p["wrong_instruction"] if wrong else p["instruction"]
        clause=p["wrong_terminal_clause"] if wrong else p["terminal_clause"]
        if not instruction or not clause:raise ValueError("missing instruction")
        result.append({"kind":label["kind"],"episode_id":str(p["episode_id"]),
                       "scene_id":p["scene_id"],"task_success":label[
                           "task_success_for_audit_only"],
                       "features":vector(instruction,clause)})
    return result


def matrix(rows: list[dict]) -> csr_matrix:
    positions=[(i,k,v) for i,row in enumerate(rows) for k,v in row[
        "features"].items()]
    rr,cc,vv=zip(*positions)
    return csr_matrix((vv,(rr,cc)),shape=(len(rows),DIM),dtype=np.float64)


def threshold(pos: np.ndarray,neg: np.ndarray) -> dict:
    choices=np.unique(np.concatenate([pos,neg]))[::-1]
    feasible=[]
    for value in np.concatenate([[choices[0]+1.],choices]):
        fpr=float(np.mean(neg>=value))
        if fpr<=.05:feasible.append((float(np.mean(pos>=value)),-fpr,float(value)))
    recall,minus_fpr,value=max(feasible)
    return {"threshold":value,"pooled_recall":recall,"pooled_fpr":-minus_fpr}


def auc(pos: np.ndarray,neg: np.ndarray) -> float:
    delta=pos[:,None]-neg[None,:]
    return float(np.mean((delta>0)+.5*(delta==0)))


def main() -> None:
    parser=argparse.ArgumentParser()
    for name in ("capture-manifest","pair-labels","verification","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args()
    capture,labels,verification=[json.loads(p.read_text()) for p in
                                 (args.capture_manifest,args.pair_labels,
                                  args.verification)]
    sha=digest(args.capture_manifest)
    if labels["capture_manifest_sha256"]!=sha or \
            verification["capture_manifest_sha256"]!=sha or \
            verification["audit_opened"] or \
            [x["records"] for x in verification["parts"]]!=[1391,351]:
        raise ValueError("unverified frozen two-view source")
    fit=load("fit",capture,labels)
    dev=load("development",capture,labels)
    if len(fit)!=3008 or len(dev)!=758:raise ValueError("changed pair counts")
    x=matrix(fit)
    y=np.array([r["kind"]=="crossing" for r in fit],dtype=np.float64)
    def objective(w):
        z=x@w
        loss=np.mean(np.logaddexp(0,z)-y*z)+.5*L2*float(w@w)
        grad=np.asarray(x.T@(expit(z)-y)).ravel()/len(y)+L2*w
        return float(loss),grad
    result=minimize(objective,np.zeros(DIM),jac=True,method="L-BFGS-B",
                    options={"maxiter":300,"ftol":1e-10})
    if not result.success or not np.all(np.isfinite(result.x)):
        raise ValueError(f"text-only fit failed: {result.message}")
    scores=matrix(dev)@result.x
    by={kind:np.array([scores[i] for i,r in enumerate(dev) if
                       r["kind"]==kind]) for kind in
        ("crossing","far_nonarrival","retreat","wrong_instruction")}
    pos=by["crossing"]
    neg=np.concatenate([by[k] for k in ("far_nonarrival","retreat",
                                        "wrong_instruction")])
    t=threshold(pos,neg)
    value=t["threshold"]
    near=np.array([scores[i] for i,r in enumerate(dev) if
                   r["kind"]=="crossing" and not r["task_success"]])
    report={
        "schema":"multiview_event_text_only_shortcut_v1",
        "capture_manifest_sha256":sha,
        "pair_labels_sha256":digest(args.pair_labels),
        "rgb_verification_sha256":digest(args.verification),
        "image_files_opened":0,"feature_dim":DIM,"l2":L2,
        "fit_pairs":len(fit),"development_pairs":len(dev),
        "optimizer_iterations":int(result.nit),
        "auc_pooled":auc(pos,neg),"threshold":t,
        "retreat_fpr":float(np.mean(by["retreat"]>=value)),
        "wrong_instruction_fpr":float(np.mean(by["wrong_instruction"]>=value)),
        "near_failure_recall":float(np.mean(near>=value)),
        "audit_opened":False,"navigation_result":False,
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()

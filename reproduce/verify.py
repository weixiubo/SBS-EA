"""Verify original frozen digests and reconstruct mean gaps from policy outcomes."""
from pathlib import Path
import hashlib
import json
import math
from statistics import mean
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.verify_anonymous_aaai_artifact import verify_final_crossed_result

def verify():
    base=ROOT/'reproduce/frozen'
    manifest=json.loads((ROOT/'reproduce/frozen_manifest.json').read_text())
    for record in manifest['entries']:
        path=base/record['path']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']:
            raise ValueError('Frozen hash mismatch: '+record['path'])
    counts=verify_final_crossed_result(base)
    folder=base/'evidence/pfsp_final_crossed_confirmation/final_analysis'
    analysis=json.loads((folder/'analysis.json').read_text())
    policies=json.loads((folder/'policy_evaluations.json').read_text())
    index={row['policy_sha256']:row for row in policies['rows']}
    observed={}
    for condition,mapping in policies['source_policy_map'].items():
        values=[]
        for digest in mapping.values():
            row=index[digest]
            values.append(mean(row['per_instance_gaps']))
        observed[condition]=mean(values)
        if not math.isclose(observed[condition],analysis['mean_gap_by_condition'][condition],abs_tol=1e-9,rel_tol=0):
            raise ValueError('Reconstructed gap mismatch: '+condition)
    return {'verified_files':len(manifest['entries']),**counts,'mean_gaps':observed,
            'strict_primary_decision':analysis['gate']['strict_primary_decision'],
            'decision':analysis['decision'],'intervals':'verified recorded values; bootstrap not recomputed by this command'}

if __name__=='__main__':print(json.dumps(verify(),indent=2))

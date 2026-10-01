"""Small deterministic PFSP search; no model, network or external benchmark."""
from dataclasses import asdict
import json
from pairwise_bpp_poc.pfsp import PFSPInstance, neh_makespan
from pairwise_bpp_poc.pfsp_source_ea import PFSPSourceEAConfig, run_pfsp_source_ea, run_pfsp_source_no_operator

def instance(seed, split):
    rows=tuple(tuple(1+((job*7+machine*5+seed)%13) for machine in range(3)) for job in range(7))
    base=PFSPInstance(family=f"family_{seed%2}",split=split,seed=seed,instance_id=f"{split}-{seed}",jobs=7,machines=3,processing_times=rows,best_known_makespan=None)
    return PFSPInstance(**{**base.__dict__,"best_known_makespan":neh_makespan(base)})

def characterize():
    dataset={s:[instance(i,s),instance(i+1,s)] for s,i in [("smoke",1),("val",3),("test",5)]}
    pool=("-completion_if_next","total_work","-completion_increase","-lookahead_pressure","head_tail_balance","work_rank - completion_rank")
    config=PFSPSourceEAConfig(population_size=12,generations=4,elite_count=2,mutation_count=6,crossover_count=3,random_count=1,max_terms=3,max_candidate_evaluations=32)
    return {f"{mode}-{seed}":asdict(fn(dataset,pool,seed=seed,config=config))
            for mode,fn in [("ea",run_pfsp_source_ea),("no_operator",run_pfsp_source_no_operator)]
            for seed in [77,101,211]}

if __name__=="__main__":
    print(json.dumps(characterize(),sort_keys=True,indent=2))

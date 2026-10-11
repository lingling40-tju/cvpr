from pathlib import Path
import importlib.util, torch
spec=importlib.util.spec_from_file_location("turn_rloo_advantage", Path(__file__).parents[1]/"verl/trainer/ppo/turn_rloo_advantage.py")
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
assert m.turn_rloo_self_check()["nonexecuted_budget_turn_identification"]
valid=torch.tensor([[1,1,0]]*4,dtype=torch.bool)
turns=torch.tensor([[0,1,-1,-1]]*4)
stop=torch.zeros_like(turns,dtype=torch.bool)
adv,_=m.turn_rloo_advantage(torch.tensor([15.,0.,0.,0.]),torch.zeros(4,3),valid,turns,stop,["g"]*4,torch.tensor([1,0,0,0],dtype=torch.bool))
assert torch.all(adv[:,2:]==0) and adv[0,:2].sum()>0 and adv[1,:2].sum()<0
print({"status":"PASS","masked_terminal_tokens_zero":True,"executed_turns_keep_outcome_credit":True})

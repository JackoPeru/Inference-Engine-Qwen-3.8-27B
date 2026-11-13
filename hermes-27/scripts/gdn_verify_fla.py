"""Verify our Gated DeltaNet recurrence vs fla naive reference (CPU)."""
import torch, numpy as np
from fla.ops.gated_delta_rule import naive_recurrent_gated_delta_rule
torch.manual_seed(0)
B,T,H,K,V = 1,8,2,8,8
q=torch.randn(B,T,H,K)*0.5; k=torch.randn(B,T,H,K)*0.5
v=torch.randn(B,T,H,V)*0.5; beta=torch.sigmoid(torch.randn(B,T,H))
g=torch.randn(B,T,H)*0.2  # log-space decay (negative-ish not required)
scale=1.0/np.sqrt(K)
o_ref, S_ref = naive_recurrent_gated_delta_rule(q,k,v,beta,g,scale=scale,output_final_state=True)
# --- our numpy recurrence ---
qq=q[0].numpy(); kk=k[0].numpy(); vv=v[0].numpy(); bb=beta[0].numpy(); gg=g[0].numpy()
S=np.zeros((H,K,V)); o=np.zeros((T,H,V))
for t in range(T):
    S = S*np.exp(gg[t])[:,None,None]
    bv = vv[t] - (S*kk[t][:,:,None]).sum(axis=1)
    bv = bv*bb[t][:,:,None] if False else bv*bb[t][:,None]
    S = S + kk[t][:,:,None]*bv[:,None,:]
    o[t] = (S* (qq[t][:,:,None]*scale)).sum(axis=1)
o_us=o_ref[0].numpy(); S_us=S_ref[0].numpy()
print("out max err:", np.abs(o-o_us).max())
print("state max err:", np.abs(S-S_us).max())
assert np.abs(o-o_us).max()<1e-4 and np.abs(S-S_us).max()<1e-4
# --- chunked == sequential: split T=8 into 3+5 carrying state ---
def run(qq,kk,vv,bb,gg,S0):
    S=S0.copy(); oo=np.zeros((qq.shape[0],H,V))
    for t in range(qq.shape[0]):
        S=S*np.exp(gg[t])[:,None,None]
        bv=vv[t]-(S*kk[t][:,:,None]).sum(axis=1); bv=bv*bb[t][:,None]
        S=S+kk[t][:,:,None]*bv[:,None,:]
        oo[t]=(S*(qq[t][:,:,None]*scale)).sum(axis=1)
    return oo,S
o1,S1=run(qq[:3],kk[:3],vv[:3],bb[:3],gg[:3],np.zeros((H,K,V)))
o2,S2=run(qq[3:],kk[3:],vv[3:],bb[3:],gg[3:],S1)
print("chunked max err:", np.abs(np.concatenate([o1,o2])-o).max(), np.abs(S2-S).max())
assert np.abs(np.concatenate([o1,o2])-o).max()<1e-5
print("GDN VERIFY PASS: recurrence==fla, chunked==sequential")

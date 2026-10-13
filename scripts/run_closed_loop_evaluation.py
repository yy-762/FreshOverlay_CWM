from __future__ import annotations

import argparse
import copy
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import _bootstrap  # noqa: F401
from cwm.checkpoint import load_checkpoint
from cwm.dynamics import compose_next_x_normalized, project_y_raw
from cwm.phy_lut import Stage1PhyLUT

IDLE_ACTION = np.array([-1.0, 0.0, -1.0], dtype=np.float32)


@dataclass
class SimConfig:
    dt: float = 0.010
    max_aoi: float = 2.0
    fresh_threshold: float = 0.100
    steps: int = 300


class ContinuousPhyLUT:
    """Stage-1 LUT with the same SINR-linear interpolation used in Stage 2."""

    def __init__(self, path: Path):
        df = pd.read_csv(path)
        self.snr = np.sort(df.SINR_dB.unique().astype(float))
        self.actions = np.unique(df[["delta_dB", "Rb_kbps", "MCS"]].to_numpy(np.float32), axis=0)
        self.host_payload_bytes = 4000
        self.passive_payload_bits = 25
        self._curves = {}
        self._feasible_cache = {}
        for action, g in df.groupby(["delta_dB", "Rb_kbps", "MCS"], sort=False):
            g = g.sort_values("SINR_dB")
            self._curves[tuple(map(float, action))] = dict(
                host=g.wifiGlaze_PER.to_numpy(float),
                passive=g.glaze_PER.to_numpy(float),
                duration=float(g.HEDataDuration_ms.iloc[0]) / 1000.0,
                nominal=float(g.nominalHost_Mbps.iloc[0]),
                embedding=bool(g.embeddingFeasible.iloc[0]),
            )
        self._delta_values = np.unique(self.actions[:, 0])
        self._rate_values = np.unique(self.actions[:, 1])
        self._mcs_values = np.unique(self.actions[:, 2])
        self._action_index = np.full(
            (len(self._delta_values), len(self._rate_values), len(self._mcs_values)), -1, dtype=np.intp
        )
        for j, action in enumerate(self.actions):
            di = np.searchsorted(self._delta_values, action[0])
            ri = np.searchsorted(self._rate_values, action[1])
            mi = np.searchsorted(self._mcs_values, action[2])
            self._action_index[di, ri, mi] = j
        curves = [self._curves[tuple(map(float, a))] for a in self.actions]
        self._host = np.stack([c["host"] for c in curves])
        self._passive = np.stack([c["passive"] for c in curves])
        self._duration = np.array([c["duration"] for c in curves])
        self._nominal = np.array([c["nominal"] for c in curves])
        self._embedding = np.array([c["embedding"] for c in curves], dtype=bool)

    def query(self, sinr, actions):
        sinr = np.asarray(sinr, float).reshape(-1)
        actions = np.asarray(actions, float).reshape(-1, 3)
        if len(actions) == 1 and len(sinr) > 1:
            actions = np.repeat(actions, len(sinr), axis=0)
        if len(sinr) == 1 and len(actions) > 1:
            sinr = np.repeat(sinr, len(actions))
        if len(sinr) != len(actions):
            raise ValueError("SINR and action batch sizes must match")
        di = np.searchsorted(self._delta_values, actions[:, 0])
        ri = np.searchsorted(self._rate_values, actions[:, 1])
        mi = np.searchsorted(self._mcs_values, actions[:, 2])
        if np.any(di >= len(self._delta_values)) or np.any(ri >= len(self._rate_values)) or np.any(mi >= len(self._mcs_values)):
            raise KeyError("Action outside the PHY lookup table")
        if (np.any(self._delta_values[di] != actions[:, 0]) or
                np.any(self._rate_values[ri] != actions[:, 1]) or
                np.any(self._mcs_values[mi] != actions[:, 2])):
            raise KeyError("Action absent from the PHY lookup table")
        idx = self._action_index[di, ri, mi]
        if np.any(idx < 0):
            raise KeyError("Action absent from the PHY lookup table")
        x = np.clip(sinr, self.snr[0], self.snr[-1])
        lo = np.clip(np.searchsorted(self.snr, x, side="right") - 1, 0, len(self.snr)-2)
        w = (x-self.snr[lo])/(self.snr[lo+1]-self.snr[lo])
        hp = np.clip(self._host[idx,lo] + w*(self._host[idx,lo+1]-self._host[idx,lo]), 0, 1)
        pp = np.clip(self._passive[idx,lo] + w*(self._passive[idx,lo+1]-self._passive[idx,lo]), 0, 1)
        dur = self._duration[idx]; nominal = self._nominal[idx]; emb = self._embedding[idx]
        feasible = (hp <= 0.1) & (((actions[:, 0] == 0)) | (emb & (pp <= 0.1)))
        return hp, pp, dur, nominal, feasible

    def feasible(self, sinr):
        # Cache candidate actions at 0.1 dB, but validate them at the
        # actual SINR before they can enter the planner.
        key = round(float(sinr), 1)
        cached = self._feasible_cache.get(key)
        if cached is None:
            *_, f = self.query(np.repeat(key, len(self.actions)), self.actions)
            cached = self.actions[f]
            self._feasible_cache[key] = cached
        if len(cached):
            *_, valid = self.query(np.repeat(float(sinr), len(cached)), cached)
            acts = cached[valid]
            if len(acts):
                return acts
        # A quantized cache can miss an action near a feasibility boundary.
        *_, valid = self.query(np.repeat(float(sinr), len(self.actions)), self.actions)
        acts = self.actions[valid]
        if len(acts):
            return acts
        return acts

    def feasible_with_quality(self, sinr):
        acts = self.feasible(sinr)
        _, pp, _, _, _ = self.query(np.repeat(float(sinr), len(acts)), acts)
        quality = ((acts[:, 0] > 0) * (1.0 - pp)).astype(np.float32)
        return acts, quality


def init_env(density, resource, seed, traffic_mode):
    rng = np.random.default_rng(seed + 1000 * resource + 17 * density)
    n = density
    rho_set = np.array([0.70, 0.90, 0.98]); sigma_set = np.array([2.0, 1.2, 0.6])
    regime = (seed + resource + density - 1) % 3
    mean = np.array([4.0, 8.0, 12.0])[resource - 1] + 2.5 * rng.standard_normal(n)
    periods = np.array([1, 2, 5])[(np.arange(1, n + 1) + seed - 1) % 3]
    return dict(n=n, resource=resource, density=density, t=0, rng=rng, mode=traffic_mode,
                rho=rho_set[regime], sigma=sigma_set[regime], mean=mean,
                sinr=mean + sigma_set[regime] * rng.standard_normal(n),
                host_rate=np.zeros(n), passive_rate=np.zeros(n), aoi=np.full(n, 0.010),
                pending=np.zeros(n, bool), latest=np.full(n, -np.inf), burst=rng.random(n) < 0.5,
                period=periods, phase=(np.arange(1, n + 1) + resource + seed - 2) % periods,
                activity=0.55 + 0.40 * rng.random(), cbr=0.1, load=0.0)


def dcf(pe_h, duration, activity):
    pe_h = np.asarray(pe_h); n = len(pe_h); w0 = 16; b = 6; retry = 7
    activity = np.clip(activity, 0.05, 1.0)
    tau = np.full(n, activity * 2 / (w0 + 1)); converged = False
    for iteration in range(1, 101):
        prod_not = np.prod(1 - tau)
        pc = 1 - prod_not / np.maximum(1 - tau, 1e-12)
        q = np.clip(1 - (1 - pc) * (1 - pe_h), 1e-6, 1 - 1e-6)
        q = np.where(np.abs(q - 0.5) < 1e-5, q + 1e-5, q)
        num = (1 - (2*q)**(b+1))*(1-q) + 2**b*(q**(b+1)-q**(retry+1))*(1-2*q)
        den = (1-2*q)*(1-q**(retry+1))
        tau_new = np.clip(activity * 2/(w0*(num/den)+1), 1e-6, 0.5)
        if np.max(np.abs(tau_new-tau)) < 1e-7:
            tau = tau_new; converged = True; break
        tau = 0.55*tau + 0.45*tau_new
    prod_not = np.prod(1-tau)
    pc = 1-prod_not/np.maximum(1-tau,1e-12)
    p_only = tau*(prod_not/np.maximum(1-tau,1e-12))
    p_host = p_only*(1-pe_h)
    p_idle = prod_not; p_collision = max(0.0, 1-p_idle-p_only.sum()); p_err = (p_only*pe_h).sum()
    t_succ = 100e-6+duration+16e-6+44e-6+34e-6
    t_mean_data = (p_only*duration).sum()/max(p_only.sum(),1e-12)
    t_coll = 100e-6+t_mean_data+88e-6
    t_slot = p_idle*9e-6+(p_host*t_succ).sum()+p_collision*t_coll+p_err*t_coll
    busy = (p_host*t_succ).sum()+p_collision*t_coll+p_err*t_coll
    return dict(p_host=p_host, pc=pc, tau=tau, t_slot=max(t_slot,1e-9), cbr=np.clip(busy/max(t_slot,1e-9),0,1), converged=converged)


def observe(env):
    onehot = np.zeros((env["n"], 3), np.float32); onehot[:, env["resource"]-1] = 1
    return np.column_stack([env["sinr"], env["host_rate"], env["passive_rate"], 1000*env["aoi"],
                            onehot, np.full(env["n"], env["cbr"]), np.full(env["n"], env["load"])]).astype(np.float32)


def step(env, actions, lut, cfg):
    n=env["n"]; rng=env["rng"]; now=env["t"]*cfg.dt; env["t"] += 1
    if env["mode"] == "periodic":
        generated = ((env["t"]-env["phase"]) % env["period"]) == 0
    else:
        turn_on=(~env["burst"]) & (rng.random(n)<0.08); turn_off=env["burst"] & (rng.random(n)<0.20)
        env["burst"][turn_on]=True; env["burst"][turn_off]=False
        generated=env["burst"] & (rng.random(n)<0.75)
    env["latest"][generated]=now; env["pending"][generated]=True
    active=actions[:,0]>=0
    hp=np.zeros(n); pp=np.zeros(n); dur=np.zeros(n); feasible=np.ones(n,bool)
    p_host=np.zeros(n)
    if active.any():
        hp[active],pp[active],dur[active],_,feasible[active]=lut.query(env["sinr"][active],actions[active])
        mac=dcf(hp[active],dur[active],env["activity"])
        p_host[active]=mac["p_host"]
    else:
        mac=dict(t_slot=9e-6,cbr=0.0)
    slots=max(cfg.dt/mac["t_slot"],1)
    host_prob=1-(1-p_host)**slots
    pass_prob=1-(1-p_host*(1-pp))**slots
    host_success=rng.random(n)<host_prob
    passive_success=env["pending"] & (rng.random(n)<pass_prob)
    host_rate=p_host*(8*lut.host_payload_bytes)/mac["t_slot"]/1e6
    passive_rate=pass_prob*env["pending"]*lut.passive_payload_bits/cfg.dt/1e3
    next_time=now+cfg.dt; aoi=np.minimum(env["aoi"]+cfg.dt,cfg.max_aoi)
    aoi[passive_success]=np.minimum(np.maximum(next_time-env["latest"][passive_success],cfg.dt),cfg.max_aoi)
    env["pending"][passive_success]=False
    env["sinr"]=env["mean"]+env["rho"]*(env["sinr"]-env["mean"])+np.sqrt(max(1-env["rho"]**2,0))*env["sigma"]*rng.standard_normal(n)
    env["activity"]=np.clip(0.92*env["activity"]+0.08*0.75+0.035*rng.standard_normal(),0.55,0.95)
    env["load"]=np.clip((env["density"]/40)*env["activity"],0,1)
    env["cbr"]=mac["cbr"]; env["host_rate"]=host_rate; env["passive_rate"]=passive_rate; env["aoi"]=aoi
    return dict(host_rate=host_rate, passive_rate=passive_rate, aoi=aoi.copy(), feasible=feasible, idle=~active,
                host_per=hp, passive_per=pp, host_success=host_success, passive_success=passive_success)


class Controllers:
    def __init__(self, model, normalizer, lut, device, beam=16,
                 host_scale_mbps=1.0, passive_scale_kbps=0.25, aoi_scale_ms=500.0):
        self.model=model.eval(); self.norm=normalizer; self.lut=lut; self.device=torch.device(device); self.beam=beam
        self.host_scale=host_scale_mbps; self.passive_scale=passive_scale_kbps; self.aoi_scale=aoi_scale_ms

    def fallback(self, sinr):
        acts=self.lut.feasible(sinr)
        return acts[0] if len(acts) else IDLE_ACTION

    def fixed(self, x):
        out=[]
        for s in x[:,0]:
            target=np.array([2,40,2],np.float32); *_,f=self.lut.query([s],[target])
            out.append(target if f[0] else self.fallback(s))
        return np.asarray(out)

    def mcs_only(self, x):
        out=[]
        for s in x[:,0]:
            acts=np.array([[2,40,m] for m in range(12)],np.float32)
            hp,pp,_,nom,f=self.lut.query(np.repeat(s,len(acts)),acts); acts=acts[f]
            if len(acts)==0: out.append(self.fallback(s)); continue
            hp,pp,_,nom,_=self.lut.query(np.repeat(s,len(acts)),acts)
            out.append(acts[np.argmax(nom*(1-hp)+0.001*acts[:,1]*(1-pp))])
        return np.asarray(out)

    @torch.inference_mode()
    def cwm(self, x, horizon, lam):
        # Batched beam search across all robots; each list item is (score,state,first_action).
        beams=[[(0.0,x[i].copy(),None)] for i in range(len(x))]
        for h in range(horizon):
            meta=[]; xb=[]; ab=[]; delivery_quality=[]
            expanded=[[] for _ in beams]
            for i,items in enumerate(beams):
                for score,state,first in items:
                    acts, qualities=self.lut.feasible_with_quality(state[0])
                    if not len(acts):
                        xn=state.copy(); xn[1:3]=0.0; xn[3]=min(xn[3]+10.0,2000.0)
                        idle_util=-lam*xn[3]/self.aoi_scale
                        expanded[i].append((score+(0.95**h)*idle_util,xn,IDLE_ACTION if first is None else first))
                        continue
                    for a, q in zip(acts, qualities):
                        # Delta=0 is a valid host-only fallback but cannot
                        # deliver the passive status stream.
                        meta.append((i,score,first,a)); xb.append(state); ab.append(a); delivery_quality.append(float(q))
            if meta:
                xt=torch.as_tensor(self.norm.normalize_x_np(np.asarray(xb)),device=self.device)
                at=torch.as_tensor(self.norm.normalize_a_np(np.asarray(ab)),device=self.device)
                y=self.model(xt,at); yr=project_y_raw(self.norm.denormalize_y_t(y))
                # Data-scale-aware utility with a passive-delivery term.
                q=torch.as_tensor(delivery_quality,device=self.device,dtype=yr.dtype)
                util=(torch.log1p(yr[:,1]/self.host_scale)
                      +0.25*torch.log1p(yr[:,2]/self.passive_scale)
                      -lam*(yr[:,3]/self.aoi_scale-0.75*q))
                xn=self.norm.denormalize_x_t(compose_next_x_normalized(y,xt,self.norm,True)).cpu().numpy()
                util=util.cpu().numpy()
                for j,(i,score,first,a) in enumerate(meta):
                    expanded[i].append((score+(0.95**h)*float(util[j]),xn[j],a.copy() if first is None else first))
            beams=[sorted(items,key=lambda z:z[0],reverse=True)[:self.beam] for items in expanded]
        return np.asarray([b[0][2] for b in beams],np.float32)


def run_episode(method, density, resource, seed, mode, ctrl, lut, cfg, lam=0.5, horizon=3, keep_aoi=False):
    env=init_env(density,resource,seed,mode); sums=[]; aois=[]; violations=0; unavoidable=0; idle_count=0; avoidable_idle=0; decisions=[]
    for _ in range(cfg.steps):
        x=observe(env); t0=time.perf_counter()
        if method=="fixed": actions=ctrl.fixed(x)
        elif method=="mcs_only": actions=ctrl.mcs_only(x)
        else: actions=ctrl.cwm(x,horizon,lam)
        decisions.append((time.perf_counter()-t0)*1000)
        r=step(env,actions,lut,cfg); sums.append(r["host_rate"].sum()+r["passive_rate"].sum()/1000)
        aois.append(1000*r["aoi"])
        idle_count += int(r["idle"].sum())
        for i in np.flatnonzero(r["idle"]):
            *_, any_valid=lut.query(np.repeat(float(x[i,0]),len(lut.actions)),lut.actions)
            avoidable_idle += int(any_valid.any())
        bad=np.flatnonzero(~r["feasible"]); violations += len(bad)
        for i in bad:
            *_, any_valid=lut.query(np.repeat(float(x[i,0]),len(lut.actions)),lut.actions)
            unavoidable += int(not any_valid.any())
    a=np.concatenate(aois)
    row=dict(method=method,lambda_a=lam,horizon=horizon,density=density,resource=resource,seed=seed,
             aggregate_throughput_mbps=float(np.mean(sums)),avg_aoi_ms=float(a.mean()),p95_aoi_ms=float(np.percentile(a,95)),
             freshness_probability=float(np.mean(a<=100)),violation_rate=float(violations/a.size),
             unavoidable_violation_rate=float(unavoidable/a.size),
             avoidable_violation_rate=float((violations-unavoidable)/a.size),
             idle_rate=float(idle_count/a.size),avoidable_idle_rate=float(avoidable_idle/a.size),
             decision_ms_mean=float(np.mean(decisions)),decision_ms_p95=float(np.percentile(decisions,95)))
    return row, a.astype(np.float32) if keep_aoi else None


def plot_results(df, aoi_store, out):
    plt.style.use("seaborn-v0_8-whitegrid")
    # Main comparison at lambda=0.5.
    comp=df[df.experiment=="comparison"]
    summary=comp.groupby("method").agg({"aggregate_throughput_mbps":["mean","std"],"avg_aoi_ms":["mean","std"],
        "p95_aoi_ms":["mean","std"],"freshness_probability":["mean","std"],"violation_rate":["mean","std"],
        "unavoidable_violation_rate":["mean","std"],"avoidable_violation_rate":["mean","std"],
        "idle_rate":["mean","std"],"avoidable_idle_rate":["mean","std"],
        "decision_ms_mean":["mean","std"]})
    summary.to_csv(out/"controller_summary.csv")
    # Pareto.
    p=df[df.experiment=="pareto"].groupby("lambda_a").agg(thpt=("aggregate_throughput_mbps","mean"),aoi=("avg_aoi_ms","mean")).reset_index()
    fig,ax=plt.subplots(figsize=(6,4.5)); ax.plot(p.aoi,p.thpt,"o-",lw=2)
    for _,r in p.iterrows(): ax.annotate(f"λ={r.lambda_a:g}",(r.aoi,r.thpt),xytext=(5,5),textcoords="offset points")
    ax.set_xlabel("Average AoI (ms)"); ax.set_ylabel("Aggregate throughput (Mbps)"); ax.set_title("CWM-H3 throughput–AoI weight sweep"); fig.tight_layout(); fig.savefig(out/"fig_pareto.png",dpi=180); plt.close(fig)
    # AoI CDF.
    fig,ax=plt.subplots(figsize=(6,4.5))
    for method,arrs in aoi_store.items():
        a=np.concatenate(arrs); xs=np.sort(a); ys=np.arange(1,len(xs)+1)/len(xs); ax.plot(xs,ys,label=method,lw=2)
    ax.set_xlabel("AoI (ms)"); ax.set_ylabel("CDF"); ax.set_title("Closed-loop AoI distribution"); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(out/"fig_aoi_cdf.png",dpi=180); plt.close(fig)
    # Density scaling.
    g=comp.groupby(["method","density"]).agg(aoi=("avg_aoi_ms","mean"),thpt=("aggregate_throughput_mbps","mean")).reset_index()
    fig,axs=plt.subplots(1,2,figsize=(11,4.3))
    for method,d in g.groupby("method"):
        axs[0].plot(d.density,d.aoi,"o-",label=method); axs[1].plot(d.density,d.thpt,"o-",label=method)
    axs[0].set(xlabel="Robots per resource",ylabel="Average AoI (ms)",title="Freshness scaling")
    axs[1].set(xlabel="Robots per resource",ylabel="Aggregate throughput (Mbps)",title="Throughput scaling")
    axs[1].legend(frameon=False); fig.tight_layout(); fig.savefig(out/"fig_density_scaling.png",dpi=180); plt.close(fig)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--quick",action="store_true"); ap.add_argument("--calibrate",action="store_true")
    ap.add_argument("--repair-plots",action="store_true",help="recreate plots from saved metrics and replayed AoI traces")
    ap.add_argument("--beam",type=int,default=None)
    ap.add_argument("--controller-only",action="store_true",help="evaluate only CWM-H3")
    ap.add_argument("--comparison-only",action="store_true",help="skip the lambda sweep")
    ap.add_argument("--output-dir",type=Path,default=None)
    ap.add_argument("--checkpoint",type=Path,default=None,help="world-model checkpoint for ablation evaluation")
    args=ap.parse_args()
    root=Path(__file__).resolve().parents[1]
    out=args.output_dir or root/"outputs"/"closed_loop"/"safe_idle_fast"
    out.mkdir(parents=True,exist_ok=True)
    lut_path=root/"data"/"stage1_phy_lut"/"phy_lut_4d.csv"; lut=ContinuousPhyLUT(lut_path)
    device="cuda" if torch.cuda.is_available() else "cpu"
    checkpoint=args.checkpoint or root/"outputs"/"default_run"/"checkpoints"/"best.pt"
    _,model,norm=load_checkpoint(checkpoint,device)
    ctrl=Controllers(model,norm,lut,device,beam=args.beam or (8 if args.quick else 16))
    if args.quick:
        cfg=SimConfig(steps=20); densities=[5]; seeds=[101]; resources=[1]
    elif args.calibrate:
        cfg=SimConfig(steps=100); densities=[5,20,40]; seeds=[101,102]; resources=[1,2,3]
    else:
        cfg=SimConfig(steps=300); densities=[5,10,20,30,40]; seeds=[101,102,103,104,105]; resources=[1,2,3]
    rows=[]; aoi_store={m:[] for m in ["fixed","mcs_only","cwm_h1","cwm_h3"]}
    comparison=[("cwm_h3",3)] if args.controller_only else [("fixed",1),("mcs_only",1),("cwm_h1",1),("cwm_h3",3)]
    if args.repair_plots:
        df=pd.read_csv(out/"episode_metrics.csv")
        if len(df)!=750:
            raise ValueError("Expected 750 completed episodes before repairing plots")
        for method,h in comparison:
            for d in densities:
                for seed in seeds:
                    for k in resources:
                        mode="bursty" if 1+(d+seed+k)%2==2 else "periodic"
                        _,a=run_episode(method,d,k,seed,mode,ctrl,lut,cfg,0.5,h,True)
                        aoi_store[method].append(a)
                        print(f"AoI replay {method}: d={d} seed={seed} k={k}",flush=True)
        plot_results(df,aoi_store,out)
        print(out)
        return
    for method,h in comparison:
        for d in densities:
            for seed in seeds:
                for k in resources:
                    mode="bursty" if 1+(d+seed+k)%2==2 else "periodic"
                    row,a=run_episode(method,d,k,seed,mode,ctrl,lut,cfg,0.5,h,True); row["experiment"]="comparison"; rows.append(row); aoi_store[method].append(a)
                    print(f"comparison {method}: d={d} seed={seed} k={k}")
    lambdas=[] if args.comparison_only else [0,0.1,0.3,1,3,10]
    for lam in lambdas:
        for d in densities:
            for seed in seeds:
                for k in resources:
                    mode="bursty" if 1+(d+seed+k)%2==2 else "periodic"
                    row,_=run_episode("cwm_h3",d,k,seed,mode,ctrl,lut,cfg,lam,3,False); row["experiment"]="pareto"; rows.append(row)
                    print(f"pareto lambda={lam:g}: d={d} seed={seed} k={k}")
    df=pd.DataFrame(rows)
    filename="quick_episode_metrics.csv" if args.quick else ("calibration_episode_metrics.csv" if args.calibrate else "episode_metrics.csv")
    df.to_csv(out/filename,index=False)
    if not args.quick and not args.calibrate and not args.controller_only: plot_results(df,aoi_store,out)
    print(out)


if __name__=="__main__": main()

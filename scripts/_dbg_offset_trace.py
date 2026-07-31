from __future__ import annotations
import logging, sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, ".")

class EventCollector(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []
    def emit(self, record):
        self.records.append({"level": record.levelname, "msg": record.getMessage()})

VIDEO_PATH = "data/frames/video_124_4min.mp4"
START_SEC = 0.0
END_SEC = 180.0
DBG_IMG_DIR = Path("data/indicators_v2/viz")
DBG_IMG_DIR.mkdir(parents=True, exist_ok=True)

def _pf(msg):
    d = {}
    try:
        side = msg[msg.index("[")+1:msg.index("]")]
        d["side"] = side
        for p in msg.split():
            if p.startswith("chain_total="): d["chain_total"] = int(p.split("=")[1])
            elif p.startswith("gen="): d["gen"] = int(p.split("=")[1])
            elif p.startswith("t="): d["t_sec"] = float(p.split("=")[1])
            elif p.startswith("score_start="): d["score_start"] = int(p.split("=")[1])
            elif p.startswith("score_after="): d["score_after"] = int(p.split("=")[1])
    except Exception: pass
    return d

def _po(msg):
    d = {}
    try:
        side = msg[msg.index("[")+1:msg.index("]")]
        d["side"] = side
        for p in msg.split():
            if p.startswith("gen="): d["gen"] = int(p.split("=")[1])
            elif p.startswith("canceled="): d["canceled"] = int(p.split("=")[1])
            elif p.startswith("surplus="): d["surplus"] = int(p.split("=")[1])
            elif p.startswith("self.forecast="): d["self_forecast"] = int(p.split("=")[1])
            elif p.startswith("other.forecast="): d["other_forecast"] = int(p.split("=")[1])
            elif p.startswith("t="): d["t_sec"] = float(p.split("=")[1])
    except Exception: pass
    return d

def main():
    col = EventCollector()
    col.setLevel(logging.INFO)
    lg = logging.getLogger("src.ojama_accounting")
    lg.setLevel(logging.INFO)
    lg.addHandler(col)
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    from src.board_state_machine import BoardState
    from src.ojama_accounting import OjamaAccountingTracker
    from src.recognition_pipeline import RecognitionPipeline
    print("Pipeline init...")
    pipe = RecognitionPipeline.load_default()
    print("  done")
    acct = OjamaAccountingTracker()
    acct.reset()
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30.0
    sf = int(START_SEC * fps)
    ef = int(END_SEC * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, sf)
    fi = sf; pp1 = None; pp2 = None; tl = []
    print("Processing {} frames...".format(ef - sf))
    while fi < ef:
        ret, frame = cap.read()
        if not ret: break
        t = fi / fps
        if frame.shape[:2] != (1080, 1920): frame = cv2.resize(frame, (1920, 1080))
        res = pipe.update(fi, t, frame)
        s1 = res.p1.state; s2 = res.p2.state
        sc1 = getattr(res.p1, "score", None); sc2 = getattr(res.p2, "score", None)
        if pp1 is not None and pp1 != s1: acct.on_state_transition("p1", pp1, s1, sc1, t)
        if pp2 is not None and pp2 != s2: acct.on_state_transition("p2", pp2, s2, sc2, t)
        if pp1 == BoardState.TSUMO_FALL and s1 == BoardState.STABLE: acct.on_tsumo_settled("p1", t)
        if pp2 == BoardState.TSUMO_FALL and s2 == BoardState.STABLE: acct.on_tsumo_settled("p2", t)
        sn = acct.get_snapshot(t)
        tl.append({"frame": fi, "t": t, "f1": sn.forecast_p1, "f2": sn.forecast_p2, "s1": str(s1), "s2": str(s2), "sc1": sc1, "sc2": sc2, "ce1": sn.chain_end_triggered_p1, "ce2": sn.chain_end_triggered_p2})
        pp1 = s1; pp2 = s2
        if (fi - sf) % 900 == 0: print("  f={} t={:.1f}s f1={} f2={}".format(fi, t, sn.forecast_p1, sn.forecast_p2))
        fi += 1
    cap.release()
    print("Done: {} frames, {} log events".format(len(tl), len(col.records)))
    fevs = []; oevs = []; bmsgs = []
    for rec in col.records:
        m = rec["msg"]
        if m.startswith("finalize["): fevs.append(_pf(m))
        elif m.startswith("offset["): oevs.append(_po(m))
        elif m.startswith("match_boundary"): bmsgs.append(m)
    print("Log: fin={} off={} bound={}".format(len(fevs), len(oevs), len(bmsgs)))
    for b in bmsgs: print("  BOUNDARY: {}".format(b))
    cev = [e for e in oevs if e.get("canceled", 0) > 0]
    print("canceled>0: {}".format(len(cev)))
    print("=== ALL OFFSET EVENTS ===")
    for e in oevs: print("  {:.2f} {} {} {} {} {} {}".format(e.get("t_sec",0),e.get("side","?"),e.get("gen",0),e.get("canceled",0),e.get("surplus",0),e.get("self_forecast",0),e.get("other_forecast",0)))
    print("")
    print("=== CANCEL FORECAST TIMELINE ===")
    for ev in cev:
        te = ev.get("t_sec", 0.0)
        side = ev.get("side", "?")
        nb = [r for r in tl if abs(r["t"] - te) <= 2.0]
        print("")
        print("--- t={:.2f}s [{}] gen={} canceled={} surplus={} ---".format(te, side, ev.get("gen",0), ev.get("canceled",0), ev.get("surplus",0)))
        pf1 = None; pf2 = None
        for r in nb:
            f1c = "*" if (pf1 is not None and r["f1"] != pf1) else " "
            f2c = "*" if (pf2 is not None and r["f2"] != pf2) else " "
            ce = ("P1END " if r["ce1"] else "") + ("P2END" if r["ce2"] else "")
            if f1c == "*" or f2c == "*" or abs(r["t"] - te) < 0.12:
                print("  {:.2f}s f1={}{} f2={}{} {} {} {}".format(r["t"],r["f1"],f1c,r["f2"],f2c,r["s1"],r["s2"],ce))
            pf1 = r["f1"]; pf2 = r["f2"]
    print("")
    print("=== FLICKER (STABLE forecast changes) ===")
    fp1 = []; fp2 = []; prev = None
    for r in tl:
        if prev is not None:
            if "STABLE" in r["s1"] and r["f1"] != prev["f1"]: fp1.append({"t": r["t"], "b": prev["f1"], "a": r["f1"], "d": r["f1"]-prev["f1"]})
            if "STABLE" in r["s2"] and r["f2"] != prev["f2"]: fp2.append({"t": r["t"], "b": prev["f2"], "a": r["f2"], "d": r["f2"]-prev["f2"]})
        prev = r
    print("  1P: {}".format(len(fp1)))
    print("  2P: {}".format(len(fp2)))
    for fl in sorted(fp1, key=lambda x:abs(x["d"]), reverse=True)[:20]: print("    1P t={:.2f}s {} -> {} {:+d}".format(fl["t"],fl["b"],fl["a"],fl["d"]))
    for fl in sorted(fp2, key=lambda x:abs(x["d"]), reverse=True)[:20]: print("    2P t={:.2f}s {} -> {} {:+d}".format(fl["t"],fl["b"],fl["a"],fl["d"]))
    print("")
    print("=== DOUBLE FINALIZE (<2.5s) ===")
    for lbl, fv in [("1P",[e for e in fevs if e.get("side")=="p1"]), ("2P",[e for e in fevs if e.get("side")=="p2"])]:
        sv = sorted(fv, key=lambda x: x.get("t_sec", 0))
        db = []
        for i in range(1, len(sv)):
            dt = sv[i].get("t_sec",0) - sv[i-1].get("t_sec",0)
            if dt < 2.5: db.append((sv[i-1].get("t_sec",0),sv[i].get("t_sec",0),dt,sv[i-1].get("chain_total",0),sv[i].get("chain_total",0)))
        print("  {}: {} pairs".format(lbl, len(db)))
        for d in db[:10]: print("    t1={:.2f}s t2={:.2f}s dt={:.3f}s chain={}->{}".format(d[0],d[1],d[2],d[3],d[4]))
    dms = [r["msg"] for r in col.records if r["msg"].startswith("tsumo_settled[")]
    p1d = sum(1 for m in dms if "p1" in m)
    p2d = sum(1 for m in dms if "p2" in m)
    print("")
    print("=== DRAIN: {} (p1={}, p2={}) ===".format(len(dms), p1d, p2d))
    for m in dms[:8]: print("  {}".format(m))
    if cev:
        cap2 = cv2.VideoCapture(VIDEO_PATH)
        fps2 = cap2.get(cv2.CAP_PROP_FPS)
        if fps2 <= 0: fps2 = 30.0
        for i, ev in enumerate(cev[:5]):
            te2 = ev.get("t_sec", 0.0)
            cap2.set(cv2.CAP_PROP_POS_FRAMES, int(te2 * fps2))
            r2, fr2 = cap2.read()
            if not r2: continue
            if fr2.shape[:2] != (1080, 1920): fr2 = cv2.resize(fr2, (1920, 1080))
            pr = fr2[0:200, :]
            fn = "_dbg_offset_{:02d}_t{:.1f}s_{}.png".format(i, te2, ev.get("side","q"))
            op = DBG_IMG_DIR / fn
            cv2.imwrite(str(op), pr)
            print("  saved: {}".format(op))
        cap2.release()
    print("=== DONE ===")

if __name__ == "__main__":
    main()

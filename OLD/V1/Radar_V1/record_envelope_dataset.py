#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import logging
import queue
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import acconeer.exptool as et


LABELS = ["none", "hold", "push", "pull", "tap", "wave"]


def now_ms() -> int:
    return int(time.time() * 1000)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def json_safe(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(x) for x in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return str(obj)


def compute_com(frame: np.ndarray) -> float:
    w = np.maximum(frame.astype(np.float32), 0.0)
    idx = np.arange(w.size, dtype=np.float32)
    denom = float(np.sum(w) + 1e-6)
    return float(np.sum(w * idx) / denom)


def bin_to_m(bin_idx: float, r_start: float, r_end: float, R: int) -> float:
    if R <= 1:
        return float(r_start)
    t = float(bin_idx) / float(R - 1)
    return float(r_start + t * (r_end - r_start))


def next_sample_index(label_dir: Path, label: str, session_id: str) -> int:
    pat = re.compile(rf"^envelope_{re.escape(label)}_{re.escape(session_id)}_(\d+)\.npz$")
    max_idx = -1
    if label_dir.exists():
        for p in label_dir.glob(f"envelope_{label}_{session_id}_*.npz"):
            m = pat.match(p.name)
            if m:
                try:
                    max_idx = max(max_idx, int(m.group(1)))
                except ValueError:
                    pass
    return max_idx + 1


def save_npz(out_path: Path, X: np.ndarray, meta: dict) -> None:
    np.savez_compressed(out_path, X=X.astype(np.float32), meta=json.dumps(json_safe(meta), ensure_ascii=False))
    print(f"\n[SAVED] {out_path}", flush=True)


def append_manifest(manifest_path: Path, meta: dict) -> None:
    with manifest_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(json_safe(meta), ensure_ascii=False) + "\n")


def parse_plan(plan: str) -> List[Tuple[str, int]]:
    out: List[Tuple[str, int]] = []
    for part in plan.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            k, v = part.split("=", 1)
        elif ":" in part:
            k, v = part.split(":", 1)
        else:
            raise ValueError(f"Plan invalid: {part!r} (folosește label=COUNT)")
        k = k.strip()
        v = v.strip()
        if k not in LABELS:
            raise ValueError(f"Label necunoscut în plan: {k!r} (permise: {LABELS})")
        n = int(v)
        if n <= 0:
            raise ValueError(f"COUNT trebuie >0 pentru {k!r}")
        out.append((k, n))
    if not out:
        raise ValueError("Plan gol")
    return out


@dataclass
class FrameItem:
    t_unix: float
    frame: np.ndarray


class RadarReader(threading.Thread):
    def __init__(self, client: Any, q: "queue.Queue[FrameItem]", stop_evt: threading.Event):
        super().__init__(daemon=True)
        self.client = client
        self.q = q
        self.stop_evt = stop_evt
        self.last_rx_time = time.time()
        self.error: Optional[str] = None
        self.frames_rx = 0

    def run(self) -> None:
        try:
            while not self.stop_evt.is_set():
                _info, data = self.client.get_next()  # blocking; de-aia e în thread
                frame = np.asarray(data).squeeze().astype(np.float32)
                if frame.ndim != 1:
                    frame = frame.reshape(-1).astype(np.float32)

                item = FrameItem(t_unix=time.time(), frame=frame)
                self.last_rx_time = item.t_unix
                self.frames_rx += 1

                try:
                    self.q.put_nowait(item)
                except queue.Full:
                    # drop oldest
                    try:
                        _ = self.q.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        self.q.put_nowait(item)
                    except queue.Full:
                        pass
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"


class TerminalUI:
    def __init__(self):
        import curses
        self.curses = curses
        self.stdscr = None
        self.has_color = False

    def __enter__(self):
        curses = self.curses
        self.stdscr = curses.initscr()
        curses.noecho()
        curses.cbreak()
        curses.curs_set(0)
        self.stdscr.nodelay(True)
        self.stdscr.keypad(True)

        if curses.has_colors():
            self.has_color = True
            curses.start_color()
            curses.use_default_colors()
            # (pair_id, fg, bg)
            curses.init_pair(1, curses.COLOR_GREEN, -1)
            curses.init_pair(2, curses.COLOR_YELLOW, -1)
            curses.init_pair(3, curses.COLOR_RED, -1)
            curses.init_pair(4, curses.COLOR_CYAN, -1)

        return self

    def __exit__(self, exc_type, exc, tb):
        curses = self.curses
        try:
            if self.stdscr is not None:
                self.stdscr.nodelay(False)
                self.stdscr.keypad(False)
        finally:
            curses.nocbreak()
            curses.echo()
            curses.endwin()

    def read_key(self) -> Optional[str]:
        if self.stdscr is None:
            return None
        try:
            ch = self.stdscr.getch()
        except Exception:
            return None
        if ch == -1:
            return None

        if ch in (ord("q"), ord("Q"), 27):  # q / ESC
            return "q"
        if ch in (ord("r"), ord("R")):
            return "r"
        if ch in (ord("p"), ord("P")):
            return "p"
        if ch in (ord("n"), ord("N")):
            return "n"
        if ch in (ord("s"), ord("S")):
            return "s"
        if ch in (ord("h"), ord("H")):
            return "h"
        return None

    def add(self, y: int, x: int, s: str, color_pair: int = 0):
        if self.stdscr is None:
            return
        try:
            if color_pair and self.has_color:
                self.stdscr.addstr(y, x, s, self.curses.color_pair(color_pair))
            else:
                self.stdscr.addstr(y, x, s)
        except Exception:
            pass

    def draw_lines(self, lines: List[Tuple[str, int]]) -> None:
        if self.stdscr is None:
            return
        self.stdscr.erase()
        max_y, max_x = self.stdscr.getmaxyx()
        for i, (s, cp) in enumerate(lines[:max_y]):
            # clip
            if len(s) > max_x - 1:
                s = s[: max_x - 1]
            self.add(i, 0, s, cp)
        try:
            self.stdscr.refresh()
        except Exception:
            pass


def clamp_int(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def make_range_gauge(r_start: float, r_end: float, peak_m: float, width: int,
                     edge_frac: float) -> Tuple[str, bool, bool]:
    """
    Returns:
      gauge_str, near_left, near_right
    """
    width = max(20, width)
    inner = width - 2
    t = (peak_m - r_start) / max(1e-9, (r_end - r_start))
    pos = int(round(t * (inner - 1)))
    pos = clamp_int(pos, 0, inner - 1)

    edge = int(round(edge_frac * inner))
    edge = max(1, edge)

    near_left = pos < edge
    near_right = pos >= (inner - edge)

    chars = ["-"] * inner
    chars[pos] = "*"

    # mark edge regions lightly
    for i in range(edge):
        if chars[i] == "-":
            chars[i] = "."
        if chars[inner - 1 - i] == "-":
            chars[inner - 1 - i] = "."

    gauge = "|" + "".join(chars) + "|"
    return gauge, near_left, near_right


def main() -> int:
    ap = argparse.ArgumentParser(description="A111 Envelope dataset recorder (terminal TUI focused on distance)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--manifest", default="manifest.jsonl")

    # recording
    ap.add_argument("--label", choices=LABELS, help="Label for manual/single-label auto")
    ap.add_argument("--auto", type=int, default=0, help="Auto record N samples for --label")
    ap.add_argument("--plan", default=None, help='Auto multi-label plan: "none=100,push=50,..."')
    ap.add_argument("--session", default=None, help="Session id (default: timestamp)")
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--update_rate", type=float, default=30.0)
    ap.add_argument("--range_start", type=float, default=0.20)
    ap.add_argument("--range_end", type=float, default=0.55)

    # behavior
    ap.add_argument("--warmup", type=float, default=0.8)
    ap.add_argument("--cooldown", type=float, default=0.15)
    ap.add_argument("--start_delay", type=float, default=0.25,
                    help="Delay after 'r' before capturing frames (s). 0 = instant")
    ap.add_argument("--edge_frac", type=float, default=0.12)

    # UI
    ap.add_argument("--ui_hz", type=float, default=12.0)
    ap.add_argument("--no_ui", action="store_true", help="No curses UI (fastest for huge batches)")
    ap.add_argument("--gauge_width", type=int, default=64)

    # pause/resume + resume from state
    ap.add_argument("--state_file", default=None,
                    help="Path to state JSON (saved continuously). Default: out_dir/run_state_<session>.json")
    ap.add_argument("--resume", default=None, help="Resume from a saved state JSON")

    args = ap.parse_args()

    # quiet logs (reduce random chatter)
    logging.getLogger("acconeer").setLevel(logging.WARNING)
    logging.getLogger("acconeer.exptool").setLevel(logging.WARNING)

    out_root = Path(args.out_dir)
    ensure_dir(out_root)
    manifest_path = out_root / args.manifest

    # ---------- load resume state (if any)
    resume_state = None
    if args.resume:
        sp = Path(args.resume)
        if not sp.exists():
            print(f"ERROR: --resume file not found: {sp}", file=sys.stderr)
            return 2
        resume_state = json.loads(sp.read_text(encoding="utf-8"))

    if resume_state:
        # pull config from state
        session_id = str(resume_state["session"])
        plan = [(l, int(n)) for l, n in resume_state["plan"]]
        remaining: Dict[str, int] = {k: int(v) for k, v in resume_state["remaining"].items()}
        cur_label_idx = int(resume_state["cur_label_idx"])
        sample_idx: Dict[str, int] = {k: int(v) for k, v in resume_state["sample_idx"].items()}

        cfg_seconds = float(resume_state["config"]["seconds"])
        cfg_update_rate = float(resume_state["config"]["update_rate"])
        cfg_r_start = float(resume_state["config"]["range_start"])
        cfg_r_end = float(resume_state["config"]["range_end"])

        # override runtime config with stored (sa nu amesteci dataset-uri)
        args.seconds = cfg_seconds
        args.update_rate = cfg_update_rate
        args.range_start = cfg_r_start
        args.range_end = cfg_r_end

        manual_mode = False
        single_label_mode = False
    else:
        session_id = args.session or time.strftime("%Y%m%d_%H%M%S")

        if args.plan:
            plan = parse_plan(args.plan)
            manual_mode = False
            single_label_mode = False
            remaining = {l: n for l, n in plan}
            cur_label_idx = 0
        else:
            if not args.label:
                print("ERROR: provide --label (manual/auto) or --plan", file=sys.stderr)
                return 2

            if args.auto > 0:
                plan = [(args.label, int(args.auto))]
                manual_mode = False
                single_label_mode = True
                remaining = {args.label: int(args.auto)}
                cur_label_idx = 0
            else:
                plan = [(args.label, 0)]  # manual infinite
                manual_mode = True
                single_label_mode = True
                remaining = {args.label: 0}
                cur_label_idx = 0

        # initialize sample_idx by scanning disk
        sample_idx = {}
        for label, _n in plan:
            d = out_root / label
            ensure_dir(d)
            sample_idx[label] = next_sample_index(d, label, session_id)

    # state file default
    if args.state_file:
        state_path = Path(args.state_file)
    else:
        state_path = out_root / f"run_state_{session_id}.json"

    # UI on/off
    wants_ui = (not args.no_ui) and sys.stdout.isatty()

    # ---------- radar setup
    cfg = et.a111.EnvelopeServiceConfig()
    cfg.update_rate = float(args.update_rate)
    cfg.range_interval = [float(args.range_start), float(args.range_end)]

    client = et.a111.Client(protocol="exploration", link="socket", host=args.host)

    print("[INFO] Connecting...", flush=True)
    try:
        connect_info = client.connect()
        session_info = client.setup_session(cfg)
        client.start_session()
    except Exception as e:
        print(f"ERROR: connect/setup/start failed: {type(e).__name__}: {e}", file=sys.stderr)
        try:
            client.disconnect()
        except Exception:
            pass
        return 1

    R = int(session_info.get("data_length", 0))
    if R <= 0:
        print("ERROR: invalid data_length from session_info", file=sys.stderr)
        try:
            client.disconnect()
        except Exception:
            pass
        return 1

    T = max(int(round(float(args.seconds) * float(args.update_rate))), 2)

    # ensure label dirs exist
    label_dirs: Dict[str, Path] = {}
    for label, _n in plan:
        d = out_root / label
        ensure_dir(d)
        label_dirs[label] = d

    # ---------- reader thread
    q: "queue.Queue[FrameItem]" = queue.Queue(maxsize=800)
    stop_evt = threading.Event()
    reader = RadarReader(client, q, stop_evt)
    reader.start()

    # warmup
    last_frame = np.zeros(R, dtype=np.float32)
    t_w0 = time.time()
    while time.time() - t_w0 < float(args.warmup):
        try:
            item = q.get(timeout=0.2)
            if item.frame.size == R:
                last_frame = item.frame
        except queue.Empty:
            pass

    # runtime
    ui_period = 1.0 / max(1.0, float(args.ui_hz))
    last_ui_t = 0.0
    last_save_t = 0.0
    last_print_t = 0.0

    # fps stats from reader
    last_rx_count = reader.frames_rx
    last_fps_t = time.time()
    fps_est = 0.0

    # noise tracking for present threshold
    noise_peaks: List[float] = []
    noise_maxlen = 120

    # state machine
    paused = False
    show_help = False
    record_requested = False
    skip_requested = False
    next_label_requested = False
    quit_requested = False

    arming = False
    arm_until_t = 0.0
    recording = False
    rec_frames: List[np.ndarray] = []

    def cur_label() -> str:
        return plan[cur_label_idx][0]

    def is_done() -> bool:
        if manual_mode:
            return False
        return all(remaining[l] <= 0 for l, _n in plan)

    def save_state():
        st = {
            "session": session_id,
            "out_dir": str(out_root),
            "plan": plan,
            "remaining": remaining,
            "cur_label_idx": cur_label_idx,
            "sample_idx": sample_idx,
            "config": {
                "seconds": float(args.seconds),
                "update_rate": float(args.update_rate),
                "range_start": float(args.range_start),
                "range_end": float(args.range_end),
            },
            "updated_unix_ms": now_ms(),
        }
        state_path.write_text(json.dumps(json_safe(st), ensure_ascii=False, indent=2), encoding="utf-8")

    def start_arm():
        nonlocal arming, arm_until_t, recording, rec_frames
        arming = True
        recording = False
        rec_frames = []
        arm_until_t = time.time() + max(0.0, float(args.start_delay))

    def abort_recording():
        nonlocal arming, recording, rec_frames
        arming = False
        recording = False
        rec_frames = []

    def finish_sample(label: str, X: np.ndarray):
        nonlocal last_save_t

        d = label_dirs[label]
        idx = sample_idx[label]
        fname = f"envelope_{label}_{session_id}_{idx:04d}.npz"
        out_path = d / fname

        peak = float(np.max(X))
        peak_bin_overall = int(np.argmax(np.max(X, axis=0)))
        peak_m_overall = bin_to_m(float(peak_bin_overall), float(args.range_start), float(args.range_end), R)

        meta = {
            "file": str(out_path),
            "label": label,
            "session": session_id,
            "sample_idx": idx,
            "created_unix_ms": now_ms(),
            "seconds": float(args.seconds),
            "update_rate": float(args.update_rate),
            "range_start": float(args.range_start),
            "range_end": float(args.range_end),
            "data_length": int(R),
            "T": int(T),
            "peak_max": peak,
            "peak_bin_over_time": peak_bin_overall,
            "peak_m_over_time": peak_m_overall,
            "host": args.host,
            "connect_info": json_safe(connect_info),
            "session_info": json_safe(session_info),
            "exptool_version": getattr(et, "__version__", None),
        }

        save_npz(out_path, X, meta)
        append_manifest(manifest_path, meta)

        sample_idx[label] += 1
        last_save_t = time.time()
        save_state()

    def advance_label():
        nonlocal cur_label_idx
        if manual_mode:
            return
        # if current label finished, go next
        while cur_label_idx < len(plan) and remaining[cur_label()] <= 0:
            if cur_label_idx + 1 < len(plan):
                cur_label_idx += 1
            else:
                break

    # ----- UI / non-UI loop
    try:
        if wants_ui:
            with TerminalUI() as ui:
                while True:
                    # keys
                    k = ui.read_key()
                    if k == "q":
                        quit_requested = True
                    elif k == "r":
                        record_requested = True
                    elif k == "p":
                        paused = not paused
                        if paused:
                            # safest: abort partial sample
                            abort_recording()
                    elif k == "s":
                        skip_requested = True
                    elif k == "n":
                        next_label_requested = True
                    elif k == "h":
                        show_help = not show_help

                    # drain queue
                    got_any = False
                    while True:
                        try:
                            item = q.get_nowait()
                        except queue.Empty:
                            break
                        if item.frame.size != R:
                            continue
                        last_frame = item.frame
                        got_any = True
                        if recording:
                            rec_frames.append(last_frame.copy())

                    now = time.time()

                    # fps estimate
                    if now - last_fps_t >= 0.5:
                        rx = reader.frames_rx
                        fps_est = (rx - last_rx_count) / (now - last_fps_t)
                        last_rx_count = rx
                        last_fps_t = now

                    # noise tracking when idle (not arming/recording)
                    if got_any and (not arming) and (not recording):
                        noise_peaks.append(float(np.max(last_frame)))
                        if len(noise_peaks) > noise_maxlen:
                            noise_peaks = noise_peaks[-noise_maxlen:]

                    noise_med = float(np.median(noise_peaks)) if len(noise_peaks) >= 20 else 0.0
                    present_thr = max(30.0, noise_med * 1.7 + 10.0)

                    # metrics
                    peak_val = float(np.max(last_frame))
                    peak_bin = int(np.argmax(last_frame))
                    peak_m = bin_to_m(float(peak_bin), float(args.range_start), float(args.range_end), R)
                    com_bin = compute_com(last_frame)
                    com_m = bin_to_m(com_bin, float(args.range_start), float(args.range_end), R)

                    gauge_w = clamp_int(int(args.gauge_width), 20, 200)
                    # adapt gauge to terminal width
                    try:
                        _h, _w = ui.stdscr.getmaxyx()
                        gauge_w = clamp_int(gauge_w, 20, max(20, _w - 2))
                    except Exception:
                        pass

                    gauge, near_left, near_right = make_range_gauge(
                        float(args.range_start), float(args.range_end), peak_m, gauge_w, float(args.edge_frac)
                    )

                    present = peak_val >= present_thr

                    # handle skip / next label
                    if skip_requested:
                        skip_requested = False
                        abort_recording()

                    if next_label_requested:
                        next_label_requested = False
                        if not manual_mode and cur_label_idx + 1 < len(plan):
                            abort_recording()
                            cur_label_idx += 1
                            advance_label()

                    # start logic
                    if quit_requested:
                        break

                    if manual_mode:
                        if record_requested and (not arming) and (not recording):
                            record_requested = False
                            start_arm()
                    else:
                        # auto/plan
                        if record_requested and (not arming) and (not recording):
                            # force one sample now (chiar dacă e pauză)
                            record_requested = False
                            start_arm()
                        elif (not paused) and (not arming) and (not recording) and (remaining[cur_label()] > 0):
                            if (now - last_save_t) >= float(args.cooldown):
                                start_arm()

                    # arming -> recording
                    if arming and now >= arm_until_t:
                        arming = False
                        recording = True
                        rec_frames = []

                    # finish recording
                    if recording and len(rec_frames) >= T:
                        X = np.stack(rec_frames[:T], axis=0)
                        finish_sample(cur_label(), X)
                        recording = False
                        rec_frames = []

                        if not manual_mode:
                            remaining[cur_label()] -= 1
                            advance_label()
                            if is_done():
                                break

                    # UI draw
                    if now - last_ui_t >= ui_period:
                        last_ui_t = now

                        # statuses & colors
                        warn_edge = near_left or near_right
                        cp_title = 4
                        cp_ok = 1
                        cp_warn = 2
                        cp_bad = 3

                        # make distance line "evident"
                        dist_line = f"PEAK: {peak_m:0.3f} m   COM: {com_m:0.3f} m   peak={peak_val:0.1f}   thr≈{present_thr:0.1f}   fps≈{fps_est:0.1f}"

                        if warn_edge:
                            edge_msg = "WARN: prea aproape de margine (mută mâna mai spre interior)"
                        else:
                            edge_msg = "OK: în interiorul ferestrei (nu ești lipit de margini)"

                        mode_msg = "MANUAL" if manual_mode else ("AUTO (plan)" if len(plan) > 1 else "AUTO")
                        lab = cur_label()
                        rem_txt = "∞" if manual_mode else str(remaining[lab])

                        state_txt = "PAUSED" if paused else "RUN"
                        rec_txt = "ARMING" if arming else ("REC" if recording else "idle")
                        rec_prog = ""
                        if arming:
                            rec_prog = f" start in {max(0.0, arm_until_t - now):0.2f}s"
                        elif recording:
                            rec_prog = f" {len(rec_frames)}/{T}"

                        header = f"A111 Envelope | {mode_msg} | label={lab} rem={rem_txt} | {state_txt} | {rec_txt}{rec_prog} | session={session_id}"

                        # choose colors
                        present_cp = cp_ok if present else 0
                        edge_cp = cp_bad if warn_edge else cp_ok
                        state_cp = cp_warn if paused else cp_ok

                        lines: List[Tuple[str, int]] = []
                        lines.append((header, cp_title))
                        lines.append((dist_line, present_cp))
                        lines.append((f"Range: {args.range_start:0.2f}m {gauge} {args.range_end:0.2f}m", edge_cp))
                        lines.append((edge_msg, edge_cp))
                        lines.append((f"Keys: r=record  p=pause/resume  s=abort/skip  n=next label  q=quit  h=help", state_cp))

                        if show_help:
                            lines.append(("", 0))
                            lines.append(("HELP:", cp_title))
                            lines.append(("- Scop: PEAK (m) să fie clar în interior și să nu apară WARN.", 0))
                            lines.append(("- Apasă r → așteaptă start_delay → apoi fă gestul în cele 2 sec.", 0))
                            lines.append(("- p pune pauză (oprește auto). q iese și salvează state pentru resume.", 0))
                            lines.append((f"- Resume: python ... --resume {state_path}", 0))

                        # reader health
                        dt_rx = now - reader.last_rx_time
                        if dt_rx > 1.0:
                            lines.append((f"WARNING: nu am primit frame-uri de {dt_rx:0.1f}s (server blocat?)", cp_bad))
                        if reader.error:
                            lines.append((f"READER ERROR: {reader.error}", cp_bad))

                        ui.draw_lines(lines)

                    time.sleep(0.002)

        else:
            # no-UI: fastest. You can still Ctrl+C / q not available.
            while True:
                now = time.time()

                # drain queue
                while True:
                    try:
                        item = q.get_nowait()
                    except queue.Empty:
                        break
                    if item.frame.size != R:
                        continue
                    last_frame = item.frame
                    if recording:
                        rec_frames.append(last_frame.copy())

                # fps estimate
                if now - last_fps_t >= 1.0:
                    rx = reader.frames_rx
                    fps_est = (rx - last_rx_count) / (now - last_fps_t)
                    last_rx_count = rx
                    last_fps_t = now

                # noise estimate
                if (not recording) and (not arming):
                    noise_peaks.append(float(np.max(last_frame)))
                    if len(noise_peaks) > noise_maxlen:
                        noise_peaks = noise_peaks[-noise_maxlen:]
                noise_med = float(np.median(noise_peaks)) if len(noise_peaks) >= 20 else 0.0
                present_thr = max(30.0, noise_med * 1.7 + 10.0)

                # auto only in no_ui mode (manual without UI not recommended)
                if manual_mode:
                    pass
                else:
                    if (not arming) and (not recording) and remaining[cur_label()] > 0 and (now - last_save_t) >= float(args.cooldown):
                        arming = True
                        arm_until_t = now + max(0.0, float(args.start_delay))
                    if arming and now >= arm_until_t:
                        arming = False
                        recording = True
                        rec_frames = []

                if recording and len(rec_frames) >= T:
                    X = np.stack(rec_frames[:T], axis=0)
                    finish_sample(cur_label(), X)
                    recording = False
                    rec_frames = []
                    if not manual_mode:
                        remaining[cur_label()] -= 1
                        advance_label()
                        if is_done():
                            break

                if now - last_print_t >= 1.0:
                    last_print_t = now
                    peak_val = float(np.max(last_frame))
                    peak_bin = int(np.argmax(last_frame))
                    peak_m = bin_to_m(float(peak_bin), float(args.range_start), float(args.range_end), R)
                    present = "YES" if peak_val >= present_thr else "no"
                    print(
                        f"\r[LIVE] fps≈{fps_est:4.1f} label={cur_label()} rem={remaining[cur_label()]:4d} "
                        f"peak@{peak_m:0.3f}m peak={peak_val:7.1f} present={present} "
                        f"state={'ARM' if arming else ('REC' if recording else 'idle')} {len(rec_frames)}/{T}   ",
                        end="",
                        flush=True,
                    )

                if reader.error:
                    print(f"\nREADER ERROR: {reader.error}", file=sys.stderr)
                    break

                time.sleep(0.005)

    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C", flush=True)
    finally:
        # save state on exit
        try:
            save_state()
            print(f"\n[INFO] State saved: {state_path}", flush=True)
        except Exception:
            pass

        stop_evt.set()
        try:
            reader.join(timeout=1.0)
        except Exception:
            pass
        try:
            client.stop_session()
        except Exception:
            pass
        try:
            client.disconnect()
        except Exception:
            pass

    print("\n[INFO] done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

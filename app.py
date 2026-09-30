"""Local Streamlit dashboard for FRC robot detection and field projection."""
from __future__ import annotations

import csv
import html
import json
import os
from pathlib import Path
from urllib.parse import quote

import cv2
import streamlit as st
from ultralytics import YOLO

from src.frcvision.geometry import (
    REEFSCAPE_FIELD_LENGTH_M,
    REEFSCAPE_FIELD_WIDTH_M,
    project_detection_rows,
    reefscape_reef_footprints,
)
from src.frcvision.video import process_video


DATA_ROOT = Path(os.getenv("FRC_DATA_ROOT", "/home/brian/Projects/RoboPerception-data"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(DATA_ROOT / "ultralytics-config"))
os.environ.setdefault("MIOPEN_DEBUG_GCN_ASM_KERNELS", "0")
os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")
ROOT = Path(__file__).resolve().parent
FIELD_ASSET_DIR = ROOT / "assets" / "field-2025"
FIELD_ASSET_CONFIG = json.loads((FIELD_ASSET_DIR / "config.json").read_text())
FIELD_ASSET_IMAGE = FIELD_ASSET_DIR / "image.png"
FIELD_PIXELS = cv2.imread(str(FIELD_ASSET_IMAGE))
if FIELD_PIXELS is None:
    raise RuntimeError(f"Unable to load field art: {FIELD_ASSET_IMAGE}")
FIELD_IMAGE_HEIGHT, FIELD_IMAGE_WIDTH = FIELD_PIXELS.shape[:2]
del FIELD_PIXELS
FIELD_CROP_TOP_LEFT = FIELD_ASSET_CONFIG["topLeft"]
FIELD_CROP_BOTTOM_RIGHT = FIELD_ASSET_CONFIG["bottomRight"]
VIDEO_DIR = DATA_ROOT / "videos"
RUN_DIR = DATA_ROOT / "runs"
OUT_DIR = DATA_ROOT / "processed"
VIDEO_BASE_URL = os.getenv(
    "FRC_VIDEO_BASE_URL", "https://llmhost2.tail72a2a1.ts.net/processed"
).rstrip("/")
FIELD_IMAGE_URL = os.getenv(
    "FRC_FIELD_IMAGE_URL", f"{VIDEO_BASE_URL.rsplit('/', 1)[0]}/field/image.png"
)
VIDEO_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)


def render_field_player(
    video_url: str,
    positions: list,
    fps: float,
    frame_count: int,
    field_length: float,
    field_width: float,
    smoothing_alpha: float,
    persistence_frames: int,
    confirmation_frames: int,
    reef_exclusion_active: bool = False,
    ocr_events: list | None = None,
) -> None:
    """Render an aspect-ratio-preserving video synchronized to the field map."""
    page = r'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<style>
*{box-sizing:border-box}body{margin:0;background:#10161d;color:#eaf0f5;font:14px system-ui,sans-serif}
.layout{display:grid;gap:14px}.panel{background:#18212a;border:1px solid #2b3946;border-radius:12px;padding:12px;min-width:0}
h3{font-size:16px;margin:0 0 9px}.sub{color:#b6c3cf;margin:0 0 9px;font-size:12px}
video{display:block;width:100%;height:auto;max-height:720px;object-fit:contain;background:#000;border-radius:8px}
.field-wrap{position:relative;width:100%;overflow:hidden;background:#222}#field-image{display:block;width:100%;height:auto}
#field-map{position:absolute;inset:0;width:100%;height:100%;overflow:visible}.reef-zone{fill:#ff6573;fill-opacity:.17;stroke:#ff6573;stroke-width:8;stroke-dasharray:18 12}
.robot{stroke:#101820;stroke-width:7}.robot-label{fill:#fff;font:bold 36px system-ui,sans-serif;text-anchor:middle;paint-order:stroke;stroke:#17212b;stroke-width:10px;stroke-linejoin:round}
.readout{display:flex;justify-content:space-between;color:#c3ced8;font-size:12px;margin-top:7px;gap:12px;flex-wrap:wrap}.credit{margin:7px 0 0;color:#9baab7;font-size:11px}
.ocr-evidence{display:flex;align-items:center;gap:12px;margin-top:10px;padding:10px;background:#10161d;border-radius:8px;min-height:82px}
.ocr-evidence img{width:180px;height:84px;object-fit:contain;background:#05080b;border-radius:5px}.ocr-copy{flex:1;min-width:0}.ocr-copy strong{display:block;font-size:14px}.ocr-copy span{display:block;color:#b6c3cf;font-size:12px;margin-top:4px}
.ocr-evidence button{border:1px solid #536577;background:#243240;color:#fff;border-radius:6px;padding:7px 10px;cursor:pointer}.ocr-evidence button[hidden]{display:none}
</style></head><body><main class="layout">
<section class="panel"><h3>Annotated match</h3><video id="match" controls playsinline preload="metadata"></video>
<div class="ocr-evidence" aria-live="polite"><img id="ocr-crop" alt="OCR crop evidence" hidden><div class="ocr-copy"><strong id="ocr-title">OCR evidence</strong><span id="ocr-detail">No OCR samples for this clip.</span></div><button id="ocr-jump" hidden>Jump to sample</button></div></section>
<section class="panel"><h3>2025 REEFSCAPE field map</h3><p class="sub" id="map-state">Clip-local track IDs; faded markers briefly remain through missed detections.</p>
<div class="field-wrap"><img id="field-image" src="__FIELD_IMAGE_URL__" alt="Top-down 2025 REEFSCAPE field diagram">
<svg id="field-map" viewBox="0 0 __IMAGE_WIDTH__ __IMAGE_HEIGHT__" preserveAspectRatio="none" role="img" aria-label="Robot detections projected onto the field"><g id="forbidden"></g><g id="robots"></g></svg></div>
<div class="readout"><span id="frame-readout">Frame —</span><span id="robot-count">0 detections</span></div>
<p class="credit">Floor-contact projections · AdvantageScope field art based on the 2025 REEFSCAPE CAD render</p></section></main>
<script>
const videoUrl=__VIDEO_URL__,positions=__POSITIONS__,ocrEvents=__OCR_EVENTS__,fps=__FPS__,frameCount=__FRAME_COUNT__;
const fieldLength=__FIELD_LENGTH__,fieldWidth=__FIELD_WIDTH__,smoothingAlpha=__SMOOTHING_ALPHA__;
const persistenceFrames=__PERSISTENCE_FRAMES__,confirmationFrames=__CONFIRMATION_FRAMES__,reefExclusionActive=__REEF_EXCLUSION_ACTIVE__;
const reefFootprints=__REEF_FOOTPRINTS__,bounds={left:__FIELD_LEFT__,top:__FIELD_TOP__,right:__FIELD_RIGHT__,bottom:__FIELD_BOTTOM__};
const video=document.getElementById('match');video.src=videoUrl;const byFrame=new Map();
for(const p of positions){const a=byFrame.get(p[0])||[];a.push(p);byFrame.set(p[0],a)}
const reef=document.getElementById('forbidden');if(reefExclusionActive)for(const poly of reefFootprints){const e=document.createElementNS('http://www.w3.org/2000/svg','polygon');e.setAttribute('class','reef-zone');e.setAttribute('points',poly.map(([x,y])=>`${bounds.right-x/fieldLength*(bounds.right-bounds.left)},${bounds.top+y/fieldWidth*(bounds.bottom-bounds.top)}`).join(' '));reef.appendChild(e)}
const group=document.getElementById('robots');let last=-1,activeOcr=null;
function drawOcr(f){let nearest=null,distance=Infinity;for(const event of ocrEvents){const d=f-event.frame;if(d>=0&&d<distance){distance=d;nearest=event}}
 const maxGap=Math.max(3,Math.round(fps*.8)),title=document.getElementById('ocr-title'),detail=document.getElementById('ocr-detail'),crop=document.getElementById('ocr-crop'),jump=document.getElementById('ocr-jump');
 if(!nearest||distance>maxGap){activeOcr=null;title.textContent='OCR evidence';detail.textContent=ocrEvents.length?`No OCR sample near frame ${f+1}; ${ocrEvents.length} saved samples.`:'No OCR samples were recorded for this clip.';crop.hidden=true;jump.hidden=true;return}
activeOcr=nearest;const seconds=(nearest.frame/fps).toFixed(2),number=nearest.number||'';
 const raw=nearest.raw_number||number,rejected=(nearest.status||'').startsWith('rejected_');
 title.textContent=nearest.status==='confirmed'?`Team ${number} confirmed`:rejected?`Rejected OCR read: ${raw||'unmatched'}`:raw&&number&&raw!==number?`Partial read ${raw} matched Team ${number}`:number?`OCR candidate: Team ${number} · unconfirmed`:nearest.status==='no_bumper_roi'?'No bumper-color panel isolated':nearest.status==='error'?'OCR backend error':'No readable digits in this crop';
detail.textContent=`${nearest.alliance||'unknown'} bumper · ${Math.round((nearest.confidence||0)*100)}% OCR confidence · track ${nearest.track_id} · ${seconds}s · ${nearest.status}`;
 if(nearest.crop_url){crop.src=nearest.crop_url;crop.hidden=false}else{crop.hidden=true}
 jump.hidden=false;jump.onclick=()=>{video.currentTime=nearest.frame/fps;video.pause();draw()};
}
function draw(){const f=Math.max(0,Math.min(frameCount-1,Math.floor((video.currentTime||0)*fps)));if(f===last)return;last=f;const robots=byFrame.get(f)||[];group.replaceChildren();
robots.forEach((p,i)=>{const x=bounds.right-p[1]/fieldLength*(bounds.right-bounds.left),y=bounds.top+p[2]/fieldWidth*(bounds.bottom-bounds.top),score=p[3],id=p.length>4?p[4]:i+1,observed=p.length<6||Boolean(p[5]),age=p.length>6?p[6]:0,number=p.length>7?p[7]:'';const color=!observed?'#d4c18d':score>=.7?'#35d7ad':score>=.5?'#f4c456':'#ff746e',opacity=observed?1:Math.max(.24,1-.72*age/Math.max(1,persistenceFrames));
const c=document.createElementNS('http://www.w3.org/2000/svg','circle');c.setAttribute('cx',x);c.setAttribute('cy',y);c.setAttribute('r',42);c.setAttribute('fill',color);c.setAttribute('class','robot');c.setAttribute('fill-opacity',opacity);c.setAttribute('stroke-opacity',opacity);const t=document.createElementNS('http://www.w3.org/2000/svg','title');t.textContent=`${number?`Team ${number} · `:''}Track ${id} · confidence ${score.toFixed(2)}`;c.appendChild(t);group.appendChild(c);
const l=document.createElementNS('http://www.w3.org/2000/svg','text');l.setAttribute('x',x);l.setAttribute('y',y+12);l.setAttribute('class','robot-label');l.setAttribute('fill-opacity',opacity);l.textContent=number?`Team ${number}`:`T${id}`;group.appendChild(l)});
document.getElementById('frame-readout').textContent=`Frame ${f+1} / ${frameCount}`;document.getElementById('robot-count').textContent=`${robots.length} tracks · ${robots.filter(p=>p.length<6||Boolean(p[5])).length} detections`;drawOcr(f)}
video.addEventListener('loadedmetadata',draw);video.addEventListener('timeupdate',draw);video.addEventListener('seeked',draw);let active=false;function tick(){if(!active)return;draw();if(video.requestVideoFrameCallback)video.requestVideoFrameCallback(tick);else setTimeout(tick,100)}video.addEventListener('play',()=>{active=true;tick()});video.addEventListener('pause',()=>{active=false;draw()});
document.getElementById('map-state').textContent=`${fieldLength.toFixed(2)} m × ${fieldWidth.toFixed(2)} m · ByteTrack · smoothing α=${smoothingAlpha.toFixed(2)} · confirmed over ${confirmationFrames} frames · holds up to ${persistenceFrames}`;
</script></body></html>'''
    replacements = {
        "__VIDEO_URL__": json.dumps(video_url),
        "__FIELD_IMAGE_URL__": html.escape(FIELD_IMAGE_URL, quote=True),
        "__IMAGE_WIDTH__": str(FIELD_IMAGE_WIDTH), "__IMAGE_HEIGHT__": str(FIELD_IMAGE_HEIGHT),
        "__FIELD_LEFT__": json.dumps(FIELD_CROP_TOP_LEFT[0]),
        "__FIELD_TOP__": json.dumps(FIELD_CROP_TOP_LEFT[1]),
        "__FIELD_RIGHT__": json.dumps(FIELD_CROP_BOTTOM_RIGHT[0]),
        "__FIELD_BOTTOM__": json.dumps(FIELD_CROP_BOTTOM_RIGHT[1]),
        "__POSITIONS__": json.dumps(positions, separators=(",", ":")),
        "__OCR_EVENTS__": json.dumps(ocr_events or [], separators=(",", ":")),
        "__FPS__": json.dumps(float(fps)), "__FRAME_COUNT__": json.dumps(int(frame_count)),
        "__FIELD_LENGTH__": json.dumps(float(field_length)), "__FIELD_WIDTH__": json.dumps(float(field_width)),
        "__SMOOTHING_ALPHA__": json.dumps(float(smoothing_alpha)),
        "__PERSISTENCE_FRAMES__": json.dumps(int(persistence_frames)),
        "__CONFIRMATION_FRAMES__": json.dumps(int(confirmation_frames)),
        "__REEF_EXCLUSION_ACTIVE__": json.dumps(bool(reef_exclusion_active)),
        "__REEF_FOOTPRINTS__": json.dumps(reefscape_reef_footprints(field_length, field_width)),
    }
    for token, value in replacements.items():
        page = page.replace(token, value)
    st.iframe(page, height="content")


st.set_page_config(page_title="FRC Robot Vision", page_icon="🤖", layout="wide")
st.title("FRC Robot Vision")
st.caption("Class-aware YOLO26 inference · ByteTrack robot IDs · smoothed field projection")

models = []
if os.getenv("FRC_MODEL"):
    models.append(Path(os.environ["FRC_MODEL"]))
models += sorted(RUN_DIR.glob("**/weights/best.pt"))
models += sorted((DATA_ROOT / "models").glob("*.pt"))
models = list(dict.fromkeys(p for p in models if p.is_file()))
choice = st.selectbox("Model checkpoint", models, format_func=str) if models else None
if not choice:
    st.info("No trained checkpoint found. Train first or set FRC_MODEL to a .pt checkpoint.")

existing = sorted(
    [*VIDEO_DIR.glob("*.mp4"), *VIDEO_DIR.glob("*.mkv"), *VIDEO_DIR.glob("*.mov")],
    key=lambda path: (
        -next((height for marker, height in (("2160p", 2160), ("1440p", 1440),
                                              ("1080p", 1080), ("720p", 720))
               if marker in path.stem.lower()), 0),
        path.name,
    ),
)
mode = st.radio("Match video", ["Upload", "Choose downloaded video"], horizontal=True)
video_path = None
if mode == "Upload":
    uploaded = st.file_uploader("Select a match video", type=["mp4", "mov", "mkv", "avi", "webm"])
    if uploaded:
        video_path = VIDEO_DIR / uploaded.name
        video_path.write_bytes(uploaded.getbuffer())
else:
    if existing:
        selected = st.selectbox("Local video", existing, format_func=lambda p: p.name)
        video_path = selected
    else:
        st.info(f"No downloaded videos found in {VIDEO_DIR}")

confidence = st.slider("Confidence threshold", .05, .95, .50, .01,
                       help="Higher values suppress weaker detections and reduce false positives, but can miss distant robots.")
col_smooth, col_confirm, col_hold = st.columns(3)
smoothing_alpha = col_smooth.slider("Track smoothing", .1, 1.0, .35, .05)
confirmation_frames = col_confirm.slider("Confirm over frames", 1, 10, 3,
    help="Require repeated detections before showing a new track.")
persistence_frames = col_hold.slider("Hold through missed frames", 0, 30, 6,
    help="Keep the last map position briefly when a tracked robot is missed.")
read_team_numbers = st.checkbox("Read team numbers from robot bumpers (OCR)", value=True,
    help="Optional asynchronous CPU OCR on robot crops.")
ocr_interval_frames = st.slider("OCR interval per track (frames)", 5, 120, 30, 5,
    disabled=not read_team_numbers)

source_calibration_path = Path(video_path).with_suffix(".calibration.json") if video_path else None
source_calibration = None
if source_calibration_path and source_calibration_path.is_file():
    try:
        source_calibration = json.loads(source_calibration_path.read_text())
    except (OSError, json.JSONDecodeError):
        st.warning("The saved source-video calibration could not be read.")
reef_calibrated = bool(source_calibration and len(source_calibration.get("image_corners", [])) == 4)
exclude_reef = st.checkbox("Reject detections centered on the 2025 Reef bases", value=False,
    disabled=not reef_calibrated, help="Uses calibrated box bottom-centers and physical Reef footprints.")
if not reef_calibrated:
    st.caption("Reef rejection is unavailable until this source video has a saved field calibration.")
device = st.selectbox(
    "Inference device", ["0", "cpu"],
    index=0 if os.getenv("FRC_DEVICE", "cpu") == "0" else 1,
    format_func=lambda value: "AMD Radeon Pro WX 9100 (GPU 0)" if value == "0" else "CPU",
    key="inference_device_cpu_default_v3",
)

if st.button("Process video", type="primary", disabled=not (video_path and choice)):
    try:
        model = YOLO(str(choice))
        output_tag = f"{Path(video_path).stem}-{Path(choice).stem}-conf{int(confidence*100):02d}-tracked-a{int(smoothing_alpha*100):02d}-h{persistence_frames:02d}-c{confirmation_frames:02d}"
        if exclude_reef: output_tag += "-reef"
        if read_team_numbers: output_tag += f"-ocr-oi{ocr_interval_frames}"
        target = OUT_DIR / f"{output_tag}.mp4"
        with st.spinner("Processing frames…"):
            summary = process_video(
                model, video_path, target, confidence=confidence,
                device=device, smoothing_alpha=smoothing_alpha,
                persistence_frames=persistence_frames, confirmation_frames=confirmation_frames,
                image_corners=source_calibration.get("image_corners") if reef_calibrated else None,
                calibration_size=(source_calibration.get("image_width"), source_calibration.get("image_height")) if reef_calibrated else None,
                exclude_reef=exclude_reef, field_length_m=REEFSCAPE_FIELD_LENGTH_M,
                field_width_m=REEFSCAPE_FIELD_WIDTH_M, source_video=video_path,
                read_team_numbers=read_team_numbers, ocr_interval_frames=ocr_interval_frames,
            )
        st.session_state["last_result"] = summary
        st.session_state["last_annotated"] = str(target)
        st.success(f"Processed {summary['frames']} frames · {summary['detections']} detections")
    except Exception as exc:
        st.error(f"Video processing failed: {exc}")

annotated_videos = sorted(OUT_DIR.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
if annotated_videos:
    preferred = st.session_state.get("last_annotated")
    default_index = next((i for i, p in enumerate(annotated_videos) if str(p) == preferred), 0)
    playback_path = st.selectbox("Annotated video", annotated_videos, index=default_index,
                                 format_func=lambda p: p.name,
                                 help="Processed videos remain available here after reruns or reloads.")
    st.subheader("Match playback and field map")
    detections_path = playback_path.with_suffix(".detections.json")
    if detections_path.is_file():
        try:
            detection_data = json.loads(detections_path.read_text())
            image_width, image_height = detection_data["width"], detection_data["height"]
            frame_count = detection_data["frames"]
            fps = detection_data.get("fps", 30)
            calibration_path = playback_path.with_suffix(".calibration.json")
            saved_calibration = json.loads(calibration_path.read_text()) if calibration_path.is_file() else None
            field_length = saved_calibration.get("field_length_m", REEFSCAPE_FIELD_LENGTH_M) if saved_calibration else REEFSCAPE_FIELD_LENGTH_M
            field_width = saved_calibration.get("field_width_m", REEFSCAPE_FIELD_WIDTH_M) if saved_calibration else REEFSCAPE_FIELD_WIDTH_M
            corners = saved_calibration.get("image_corners") if saved_calibration else [[0, 0], [image_width, 0], [image_width, image_height], [0, image_height]]
            projected_positions = project_detection_rows(
                detection_data.get("positions", []), corners, field_length, field_width,
                exclude_reef=bool(detection_data.get("reef_exclusion")),
            )
            map_reef_calibrated = bool(saved_calibration)
            exclude_map_reef = st.checkbox(
                "Hide projected detections on the Reef bases", value=False,
                disabled=not map_reef_calibrated, key=f"map_reef_{playback_path.name}",
            )
            if exclude_map_reef and map_reef_calibrated:
                projected_positions = project_detection_rows(
                    detection_data.get("positions", []), corners, field_length,
                    field_width, exclude_reef=True,
                )
            relative_video_url = f"{VIDEO_BASE_URL}/{quote(playback_path.name)}"
            ocr_events = detection_data.get(
                "ocr_events", detection_data.get("number_reads", []),
            )
            playback_ocr_events = []
            for event in ocr_events:
                event = dict(event)
                crop_file = event.get("crop_file")
                if crop_file:
                    event["crop_url"] = f"{VIDEO_BASE_URL}/{quote(crop_file, safe='/')}"
                playback_ocr_events.append(event)
            render_field_player(
                relative_video_url, projected_positions, fps, frame_count, field_length,
                field_width, detection_data.get("smoothing_alpha", .35),
                detection_data.get("persistence_frames", 6), detection_data.get("confirmation_frames", 3),
                bool(detection_data.get("reef_exclusion")),
                playback_ocr_events,
            )
            if not saved_calibration:
                st.warning("Approximate projection: the full image is currently treated as the field. Calibrate the carpet corners before using positions.")
            metrics_path = playback_path.with_suffix(".csv")
            rows = list(csv.DictReader(metrics_path.open(newline=""))) if metrics_path.is_file() else []
            latencies = [float(row["inference_ms"]) for row in rows if row.get("inference_ms")]
            processing_ms = [float(row["processing_ms"]) for row in rows if row.get("processing_ms")]
            summary = st.session_state.get("last_result", {})
            total_detections = sum(int(row["detections"]) for row in rows) if rows else 0
            details = st.columns(4)
            details[0].metric("Frames", f"{frame_count:,}")
            details[1].metric("Detections", f"{total_detections:,}")
            details[2].metric("YOLO latency", f"{sum(latencies)/len(latencies):.1f} ms/frame" if latencies else "—")
            details[3].metric("Processing throughput", f"{1000/(sum(processing_ms)/len(processing_ms)):.1f} FPS" if processing_ms and sum(processing_ms) else "—")
            st.caption("End-to-end offline clip throughput; real-time processing requires matching or exceeding source FPS.")
            counts = detection_data.get("class_counts", {})
            if counts:
                st.write("Detections by class (frame-level): " + " · ".join(f"{html.escape(name)}: {count:,}" for name, count in counts.items()))
            match_roster = detection_data.get("match_roster")
            if match_roster:
                alliances = match_roster["alliance_teams"]
                st.caption(
                    f"{match_roster.get('event', '')} · {match_roster.get('match', '')} — "
                    f"Red: {', '.join(alliances['red'])} · Blue: {', '.join(alliances['blue'])} · "
                    f"roster source: {match_roster.get('roster_source', 'recorded with video')}",
                )
            if detection_data.get("team_number_ocr"):
                confirmed_numbers = detection_data.get("team_numbers", [])
                candidates = [event for event in ocr_events if event.get("number")]
                unique_numbers = sorted({str(item['number']) for item in confirmed_numbers},
                                        key=lambda value: int(value))
                rejected_reads = sum(str(event.get('status', '')).startswith('rejected_')
                                     for event in ocr_events)
                st.write(
                    f"OCR: {len(unique_numbers)} roster team numbers across {len(confirmed_numbers)} "
                    f"confirmed tracks · {len(candidates)} raw reads · {rejected_reads} rejected "
                    f"· {detection_data.get('ocr_partial_matches', 0)} partial matches · "
                    f"{detection_data.get('ocr_mean_ms', 0):.1f} ms/crop average"
                )
                if confirmed_numbers:
                    st.success("Confirmed team numbers: " + " · ".join(unique_numbers))
                    st.caption("Track assignments: " + " · ".join(
                        f"T{item['track_id']} → Team {item['number']} ({item['confidence']:.0%})"
                        for item in confirmed_numbers
                    ))
                else:
                    st.caption("No team number has repeated consistently enough to mark as confirmed.")
                with st.expander("Review OCR crop evidence", expanded=bool(candidates)):
                    if not ocr_events:
                        st.info("No bumper crops were sampled for this clip.")
                    else:
                        def format_ocr_event(index):
                            event = ocr_events[index]
                            number = event.get("number") or "no digits"
                            raw = event.get("raw_number", number)
                            readout = f"{raw} → {number}" if number != "no digits" and raw != number else number
                            time_s = event.get("frame", 0) / max(float(fps), 1.0)
                            return (
                                f"{time_s:6.2f}s · T{event.get('track_id', '?')} · "
                                f"{readout} · {event.get('status', 'sample')}"
                            )
                        event_index = st.selectbox(
                            "Sample", range(len(ocr_events)), format_func=format_ocr_event,
                            key=f"ocr_review_{playback_path.name}",
                        )
                        event = ocr_events[event_index]
                        evidence_columns = st.columns([1, 2])
                        crop_file = event.get("crop_file")
                        evidence_path = (OUT_DIR / crop_file).resolve() if crop_file else None
                        if (evidence_path and evidence_path.is_relative_to(OUT_DIR.resolve())
                                and evidence_path.is_file()):
                            evidence_columns[0].image(
                                str(evidence_path), caption="Saved bumper crop", width="stretch",
                            )
                        else:
                            evidence_columns[0].info("This older result has no saved crop image.")
                        event_time = event.get("frame", 0) / max(float(fps), 1.0)
                        evidence_columns[1].write({
                            "time_seconds": round(event_time, 2),
                            "track_id": event.get("track_id"),
                            "bumper_color": event.get("alliance", "unknown"),
                            "raw_text": event.get("raw_number", event.get("number", "")),
                            "matched_team": event.get("number", "") if event.get("match_type") in {"exact", "partial"} else "",
                            "match_type": event.get("match_type", "unvalidated"),
                            "ocr_confidence": event.get("confidence", 0),
                            "status": event.get("status", "legacy read"),
                            "repeated_read_confirmed": event.get("confirmed", False),
                        })
            if metrics_path.is_file():
                st.download_button("Download per-frame metrics CSV", metrics_path.read_bytes(), file_name=metrics_path.name, mime="text/csv")
        except Exception as exc:
            st.error(f"Could not load playback metadata: {exc}")
    else:
        st.video(str(playback_path))
else:
    st.info("Process a match video to create an annotated MP4. It will appear here for playback.")

if video_path and Path(video_path).is_file():
    st.subheader("Calibrate the field projection")
    st.caption("Select the field-art corners in order: red-top, blue-top, blue-bottom, red-bottom. Calibrate a stable wide shot; cuts and zooms need separate calibrations.")
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ref_frame = st.slider("Calibration reference frame", 0, max(0, total - 1), min(total // 2, max(0, total - 1)), key=f"calibration_frame_{Path(video_path).name}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, ref_frame)
    ok, calibration_image = cap.read(); cap.release()
    if ok:
        ih, iw = calibration_image.shape[:2]
        st.image(cv2.cvtColor(calibration_image, cv2.COLOR_BGR2RGB), caption=f"Calibration image coordinates: {iw} × {ih} px", width="stretch")
        field_length = st.number_input("Field length (m)", min_value=.01, max_value=60.0, value=float(REEFSCAPE_FIELD_LENGTH_M), key=f"field_length_{Path(video_path).name}")
        field_width = st.number_input("Field width (m)", min_value=.01, max_value=30.0, value=float(REEFSCAPE_FIELD_WIDTH_M), key=f"field_width_{Path(video_path).name}")
        labels = ["Red alliance · top corner", "Blue alliance · top corner", "Blue alliance · bottom corner", "Red alliance · bottom corner"]
        defaults = [(0, 0), (iw, 0), (iw, ih), (0, ih)]
        if source_calibration:
            sx, sy = iw/source_calibration["image_width"], ih/source_calibration["image_height"]
            defaults = [(round(x*sx), round(y*sy)) for x,y in source_calibration["image_corners"]]
        image_corners = []
        for pair_start in range(0, 4, 2):
            corner_columns = st.columns(2)
            for offset, column in enumerate(corner_columns):
                point_idx = pair_start + offset
                x_value, y_value = defaults[point_idx]
                with column:
                    st.markdown(f"**{labels[point_idx]} carpet corner**")
                    coord_x = st.number_input("x (px)", 0, iw, int(x_value), key=f"cal_{Path(video_path).name}_{point_idx}_x")
                    coord_y = st.number_input("y (px)", 0, ih, int(y_value), key=f"cal_{Path(video_path).name}_{point_idx}_y")
                image_corners.append([coord_x, coord_y])
        if st.button("Save calibration for this video", key=f"save_calibration_{Path(video_path).name}"):
            calibration_record = {"source_video": str(Path(video_path).resolve()), "image_corners": image_corners,
                "image_width": iw, "image_height": ih, "field_length_m": field_length, "field_width_m": field_width}
            Path(video_path).with_suffix(".calibration.json").write_text(json.dumps(calibration_record, indent=2))
            st.success("Calibration saved with this source video. Re-select it to apply Reef rejection.")

"""
dashboard.py - lightweight web dashboard for autopilot status

serves a single page on port 8080 that shows the current state
of the publishing pipeline: phase, upload progress, schedule
timeline, and recent activity.

auto-refreshes every 60 seconds. no frameworks, no deps beyond
the standard library.

usage:
    python3 tools/dashboard.py --project first-blend-test
    python3 tools/dashboard.py --project first-blend-test --port 80

runs as a systemd service on the VPS. access at http://SERVER_IP:8080
"""

import os
import sys
import json
import glob
import subprocess
import argparse
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def get_video_duration(filepath):
    """ask ffprobe how long a video is."""
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', filepath],
            capture_output=True, text=True
        )
        info = json.loads(result.stdout)
        return float(info.get('format', {}).get('duration', 0))
    except Exception:
        return 0


def gather_stats(project_dir):
    """pull together everything the dashboard needs to display."""
    output_dir = os.path.join(project_dir, 'output')
    schedule_path = os.path.join(output_dir, 'schedule.json')
    state_path = os.path.join(project_dir, 'autopilot-state.json')
    rhythm_path = os.path.join(project_dir, 'rhythm.json')
    log_path = '/var/log/autopilot.log'

    stats = {
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'phase': 'unknown',
        'phase_entered': '',
        'next_burst': '',
        'uploaded': [],
        'pending': [],
        'failed': [],
        'total_videos_on_disk': 0,
        'unscheduled': 0,
        'rhythm': {},
        'recent_log': [],
        'schedule_span': '',
    }

    # autopilot state
    if os.path.exists(state_path):
        with open(state_path) as f:
            state = json.load(f)
        stats['phase'] = state.get('phase', 'unknown')
        stats['phase_entered'] = state.get('phase_entered', '')[:19]
        stats['next_burst'] = state.get('next_burst', '')[:10]

    # rhythm config
    if os.path.exists(rhythm_path):
        with open(rhythm_path) as f:
            stats['rhythm'] = json.load(f)

    # schedule
    if os.path.exists(schedule_path):
        with open(schedule_path) as f:
            schedule = json.load(f)

        for e in schedule:
            status = e.get('status', 'pending')
            entry = {
                'title': e.get('title', '?'),
                'filename': e.get('filename', '?'),
                'publish_at': e.get('publish_at', ''),
                'video_id': e.get('video_id', ''),
                'uploaded_at': e.get('uploaded_at', ''),
            }
            if status == 'uploaded':
                stats['uploaded'].append(entry)
            elif status == 'pending':
                stats['pending'].append(entry)
            elif status == 'failed':
                stats['failed'].append(entry)

        if schedule:
            dates = [e['publish_at'][:10] for e in schedule if e.get('publish_at')]
            if dates:
                stats['schedule_span'] = f"{min(dates)} to {max(dates)}"

    # videos on disk
    mp4s = glob.glob(os.path.join(output_dir, '*.mp4'))
    stats['total_videos_on_disk'] = len(mp4s)

    scheduled_files = set()
    if os.path.exists(schedule_path):
        with open(schedule_path) as f:
            entries = json.load(f)
        scheduled_files = set(e['filename'] for e in entries)
    stats['unscheduled'] = len([f for f in mp4s if os.path.basename(f) not in scheduled_files])

    # recent log lines
    if os.path.exists(log_path):
        try:
            with open(log_path) as f:
                lines = f.readlines()
            stats['recent_log'] = [l.strip() for l in lines[-20:]]
        except Exception:
            pass

    return stats


def render_html(stats):
    """generate the dashboard page."""

    phase_colors = {
        'quiet': '#6b7280',
        'rendering': '#f59e0b',
        'scheduling': '#3b82f6',
        'uploading': '#10b981',
        'unknown': '#ef4444',
    }
    phase_color = phase_colors.get(stats['phase'], '#6b7280')

    # build uploaded rows (most recent first)
    uploaded_rows = ''
    for e in reversed(stats['uploaded'][-20:]):
        yt_link = f'<a href="https://youtube.com/watch?v={e["video_id"]}" target="_blank">{e["video_id"]}</a>' if e['video_id'] else ''
        uploaded_rows += f'''<tr>
            <td>{e['title']}</td>
            <td>{e['publish_at'][:16]}</td>
            <td>{yt_link}</td>
        </tr>'''

    # build pending rows
    pending_rows = ''
    for e in stats['pending'][:15]:
        pending_rows += f'''<tr>
            <td>{e['title']}</td>
            <td>{e['publish_at'][:16]}</td>
            <td>{e['filename']}</td>
        </tr>'''

    # build log
    log_lines = '\n'.join(stats['recent_log']) if stats['recent_log'] else 'no log yet'

    # rhythm summary
    rhythm = stats.get('rhythm', {})
    rhythm_html = ''
    if rhythm:
        rhythm_html = f"""
        <div class="card">
            <h2>rhythm</h2>
            <div class="stats-grid">
                <div class="stat"><span class="label">burst size</span><span class="value">{rhythm.get('burst_size', '?')}</span></div>
                <div class="stat"><span class="label">burst density</span><span class="value">{rhythm.get('burst_density', '?')}/day</span></div>
                <div class="stat"><span class="label">cooldown</span><span class="value">{rhythm.get('cooldown_days', '?')} days</span></div>
                <div class="stat"><span class="label">solo post chance</span><span class="value">{int(rhythm.get('solo_post_chance', 0)*100)}%/day</span></div>
                <div class="stat"><span class="label">upload limit</span><span class="value">{rhythm.get('uploads_per_day', 6)}/day</span></div>
                <div class="stat"><span class="label">preset</span><span class="value">{rhythm.get('render_preset', '?')}</span></div>
            </div>
        </div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="60">
<title>collide-o-scope</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
        font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace;
        background: #0a0a0a;
        color: #d4d4d4;
        padding: 2rem;
        max-width: 900px;
        margin: 0 auto;
        font-size: 13px;
        line-height: 1.6;
    }}
    h1 {{
        font-size: 1.1rem;
        font-weight: 400;
        color: #737373;
        margin-bottom: 2rem;
        letter-spacing: 0.05em;
    }}
    h1 span {{ color: #e5e5e5; }}
    h2 {{
        font-size: 0.85rem;
        font-weight: 400;
        color: #525252;
        text-transform: lowercase;
        margin-bottom: 1rem;
        letter-spacing: 0.08em;
    }}
    .card {{
        background: #141414;
        border: 1px solid #1f1f1f;
        border-radius: 6px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }}
    .phase-badge {{
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 4px;
        font-size: 0.9rem;
        font-weight: 500;
        color: #0a0a0a;
        background: {phase_color};
        letter-spacing: 0.05em;
    }}
    .stats-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
        gap: 1rem;
        margin-top: 1rem;
    }}
    .stat {{
        display: flex;
        flex-direction: column;
        gap: 0.2rem;
    }}
    .stat .label {{
        color: #525252;
        font-size: 0.75rem;
        text-transform: lowercase;
    }}
    .stat .value {{
        color: #e5e5e5;
        font-size: 1.1rem;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        margin-top: 0.5rem;
    }}
    th {{
        text-align: left;
        color: #525252;
        font-weight: 400;
        font-size: 0.75rem;
        padding: 0.5rem 0.5rem;
        border-bottom: 1px solid #1f1f1f;
        text-transform: lowercase;
    }}
    td {{
        padding: 0.4rem 0.5rem;
        border-bottom: 1px solid #111;
        color: #a3a3a3;
        font-size: 0.8rem;
    }}
    a {{
        color: #6b7280;
        text-decoration: none;
    }}
    a:hover {{ color: #e5e5e5; }}
    pre {{
        background: #111;
        border: 1px solid #1a1a1a;
        border-radius: 4px;
        padding: 1rem;
        overflow-x: auto;
        font-size: 0.75rem;
        color: #737373;
        line-height: 1.5;
        max-height: 300px;
        overflow-y: auto;
    }}
    .meta {{
        color: #404040;
        font-size: 0.7rem;
        margin-top: 2rem;
        text-align: right;
    }}
</style>
</head>
<body>

<h1><span>collide-o-scope</span> autopilot</h1>

<div class="card">
    <h2>status</h2>
    <div class="phase-badge">{stats['phase']}</div>
    <div class="stats-grid">
        <div class="stat"><span class="label">uploaded</span><span class="value">{len(stats['uploaded'])}</span></div>
        <div class="stat"><span class="label">pending</span><span class="value">{len(stats['pending'])}</span></div>
        <div class="stat"><span class="label">failed</span><span class="value">{len(stats['failed'])}</span></div>
        <div class="stat"><span class="label">on disk</span><span class="value">{stats['total_videos_on_disk']}</span></div>
        <div class="stat"><span class="label">unscheduled</span><span class="value">{stats['unscheduled']}</span></div>
        <div class="stat"><span class="label">schedule span</span><span class="value">{stats['schedule_span']}</span></div>
        <div class="stat"><span class="label">next burst</span><span class="value">{stats['next_burst'] or 'n/a'}</span></div>
        <div class="stat"><span class="label">phase since</span><span class="value">{stats['phase_entered'][:10]}</span></div>
    </div>
</div>

{rhythm_html}

<div class="card">
    <h2>pending ({len(stats['pending'])})</h2>
    <table>
        <tr><th>title</th><th>publish at</th><th>file</th></tr>
        {pending_rows if pending_rows else '<tr><td colspan="3">nothing pending</td></tr>'}
    </table>
</div>

<div class="card">
    <h2>recent uploads ({len(stats['uploaded'])})</h2>
    <table>
        <tr><th>title</th><th>published</th><th>youtube</th></tr>
        {uploaded_rows if uploaded_rows else '<tr><td colspan="3">nothing uploaded yet</td></tr>'}
    </table>
</div>

<div class="card">
    <h2>log</h2>
    <pre>{log_lines}</pre>
</div>

<div class="meta">refreshes every 60s &middot; {stats['generated_at']}</div>

</body>
</html>"""


class DashboardHandler(BaseHTTPRequestHandler):
    """serve the dashboard on every request."""

    project_dir = None

    def do_GET(self):
        stats = gather_stats(self.project_dir)
        html = render_html(stats)
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(html.encode())

    def log_message(self, format, *args):
        # quiet — don't spam stdout with every request
        pass


def main():
    parser = argparse.ArgumentParser(description='autopilot status dashboard')
    parser.add_argument('--project', required=True, help='project name')
    parser.add_argument('--port', type=int, default=8080, help='port to serve on (default 8080)')
    args = parser.parse_args()

    project_dir = os.path.join(PROJECT_ROOT, 'projects', args.project)
    if not os.path.exists(project_dir):
        print(f"error: project not found: {project_dir}")
        sys.exit(1)

    DashboardHandler.project_dir = project_dir

    server = HTTPServer(('0.0.0.0', args.port), DashboardHandler)
    print(f"dashboard running on http://0.0.0.0:{args.port}")
    print(f"project: {args.project}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
        server.server_close()


if __name__ == '__main__':
    main()

#!/bin/bash
# setup-server.sh - provision a fresh VPS for autopilot
#
# run this on a fresh ubuntu 24.04 VPS (hetzner CX22 recommended, ~$4/mo).
# does everything except the OAuth browser flow, which you handle after.
#
# usage:
#   scp tools/setup-server.sh user@server:~/
#   ssh user@server
#   chmod +x setup-server.sh
#   ./setup-server.sh
#
# after running this script:
#   1. copy credentials:
#      scp client_secret.json user@server:~/collide-o-scope/
#      scp token.json user@server:~/collide-o-scope/
#
#   2. sync the video library:
#      rsync -avz --progress library/video/ user@server:~/collide-o-scope/library/video/
#
#   3. sync project outputs (if you want existing videos on the server):
#      rsync -avz --progress projects/ user@server:~/collide-o-scope/projects/
#
#   4. test the autopilot:
#      python3 tools/autopilot.py --project first-blend-test --status
#      python3 tools/autopilot.py --project first-blend-test --dry-run
#
#   5. enable the cron job (uncomment the last line in this script or add manually)

set -e

echo "=== system packages ==="
sudo apt update
sudo apt install -y python3 python3-pip python3-venv ffmpeg git

echo ""
echo "=== clone repo ==="
cd ~
if [ ! -d "collide-o-scope" ]; then
    git clone https://github.com/luisqueral/collide-o-scope.git
    cd collide-o-scope
else
    cd collide-o-scope
    git pull
fi

echo ""
echo "=== python dependencies ==="
pip3 install --user -r requirements.txt
pip3 install --user google-api-python-client google-auth-oauthlib

echo ""
echo "=== create directories ==="
mkdir -p library/video
mkdir -p library/audio
mkdir -p projects/first-blend-test/output

echo ""
echo "=== verify ==="
python3 --version
ffmpeg -version | head -1
echo ""
echo "checking python imports..."
python3 -c "
import numpy; print(f'  numpy {numpy.__version__}')
from PIL import Image; print(f'  Pillow ok')
import sklearn; print(f'  scikit-learn {sklearn.__version__}')
from googleapiclient.discovery import build; print(f'  google-api-python-client ok')
from google_auth_oauthlib.flow import InstalledAppFlow; print(f'  google-auth-oauthlib ok')
"

echo ""
echo "=== cron setup ==="
# write the cron job but don't enable it yet — user should test first
CRON_LINE="0 * * * * cd $HOME/collide-o-scope && python3 tools/autopilot.py --project first-blend-test >> /var/log/autopilot.log 2>&1"
echo "to enable autopilot cron, run:"
echo "  (crontab -l 2>/dev/null; echo '$CRON_LINE') | crontab -"
echo ""
echo "or to test manually first:"
echo "  cd ~/collide-o-scope"
echo "  python3 tools/autopilot.py --project first-blend-test --status"
echo "  python3 tools/autopilot.py --project first-blend-test --dry-run"

echo ""
echo "=== done ==="
echo ""
echo "next steps:"
echo "  1. scp client_secret.json and token.json from your local machine"
echo "  2. rsync library/video/ from your local machine"
echo "  3. rsync projects/ if you want existing outputs on the server"
echo "  4. test with --status and --dry-run"
echo "  5. enable the cron job"

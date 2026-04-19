# RunPod/Linux CPU Runbook

Use this flow for Linux CPU pods with a persistent `/workspace` volume.

## One-Time Setup

```bash
cd /workspace
git clone <your-repo-url> magnate
cd /workspace/magnate

apt-get update
apt-get install -y curl ca-certificates gnupg
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt-get install -y nodejs python3.12 python3.12-venv
npm install -g yarn
npm install -g npm@11.11.0

yarn install

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

export TMPDIR=/workspace/tmp
mkdir -p /workspace/tmp

python -m pip install --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple \
  -r requirements.txt
```

## Each New Pod Session

```bash
cd /workspace/magnate
git pull --ff-only
source .venv/bin/activate
export TMPDIR=/workspace/tmp
npm install -g yarn
yarn install
```

For long runs, prefer `tmux`.

## macOS/Linux Local Python Setup

Manual equivalent when not using the Windows setup script:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple \
  -r requirements-dev.txt
```

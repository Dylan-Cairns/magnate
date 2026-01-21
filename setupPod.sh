set -euo pipefail

apt-get update
apt-get install -y curl ca-certificates gnupg
curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
apt-get update
apt-get install -y nodejs python3.11 python3.11-venv
if ! command -v npm >/dev/null 2>&1; then
  echo "npm not found after nodejs install; ensure NodeSource setup succeeded." >&2
  exit 1
fi
if ! node -v | grep -q '^v22\.'; then
  echo "Expected Node.js 22.x but found: $(node -v)" >&2
  exit 1
fi
npm install -g yarn
npm install -g npm@11.11.0

GIT_EMAIL="${RUNPOD_SECRET_dylan_email:-}"

mkdir -p /workspace/.ssh
chmod 700 /workspace/.ssh

mkdir -p /root
ln -sfn /workspace/.ssh /root/.ssh
chmod 700 /root/.ssh

if [ -f /workspace/.ssh/id_ed25519_github ]; then
  chmod 600 /workspace/.ssh/id_ed25519_github
fi

touch /root/.ssh/known_hosts
ssh-keyscan github.com >> /root/.ssh/known_hosts 2>/dev/null || true

cat > /root/.ssh/config <<'EOF'
Host github.com
  IdentityFile /root/.ssh/id_ed25519_github
  IdentitiesOnly yes
EOF
chmod 600 /root/.ssh/config

git config --file /workspace/.gitconfig user.name "Dylan"
git config --file /workspace/.gitconfig user.email "${GIT_EMAIL}"
git config --file /workspace/.gitconfig init.defaultBranch main
ln -sfn /workspace/.gitconfig /root/.gitconfig

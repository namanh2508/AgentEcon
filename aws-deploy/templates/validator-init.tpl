#!/bin/bash
# AgentEcon Validator Node Initialization Script

set -e

PEER_INDEX=${peer_index}
PROJECT_NAME=${project_name}
LOGICAL_VALIDATORS_PER_PEER=${logical_validators_per_peer}

echo "Initializing AgentEcon Validator $PEER_INDEX..."

# Update system
apt-get update -y
apt-get upgrade -y

# Install prerequisites
apt-get install -y \
    curl \
    wget \
    git \
    build-essential \
    jq \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release

# Install Docker
if ! command -v docker &> /dev/null; then
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
    apt-get update -y
    apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
fi

# Install Go 1.22
if ! command -v go version &> /dev/null || ! go version | grep -q "go1.22"; then
    wget https://go.dev/dl/go1.22.0.linux-amd64.tar.gz
    tar -C /usr/local -xzf go1.22.0.linux-amd64.tar.gz
    rm go1.22.0.linux-amd64.tar.gz
fi

export PATH=$PATH:/usr/local/go/bin
export GOPATH=$HOME/go
export PATH=$PATH:$GOPATH/bin

# Create agentecon user
useradd -m -s /bin/bash agentecon || true
usermod -aG docker agentecon

# Clone or update AgentEcon repository
mkdir -p $GOPATH/src/github.com/agentecon
if [ ! -d "$GOPATH/src/github.com/agentecon/AgentEcon" ]; then
    git clone https://github.com/anonymous/agentecon.git $GOPATH/src/github.com/agentecon/AgentEcon
fi

cd $GOPATH/src/github.com/agentecon/AgentEcon
git pull origin main

# Setup Fabric
cd chaincode
go mod download
go build -o agentecon .

# Initialize Fabric peer
mkdir -p /opt/fabric
export FABRIC_CFG_PATH=/opt/fabric/config

# Create peer configuration
cat > /opt/fabric/config/core.yaml <<EOF
peer:
  id: validator$PEER_INDEX
  networkId: dev
  listenAddress: 0.0.0.0:7051
  address: 0.0.0.0:7051
  chaincodeAddress: 0.0.0.0:7052
  chaincodeListenAddress: 0.0.0.0:7052
  addressAutoDetect: true
  gossip:
    bootstrap: 10.0.1.10:7051
    externalEndpoint: 10.0.1.$PEER_INDEX:7051
  tls:
    enabled: true
    cert:
      file: /etc/hyperledger/fabric/tls/server.crt
    key:
      file: /etc/hyperledger/fabric/tls/server.key
    rootcert:
      file: /etc/hyperledger/fabric/tls/ca.crt
  genesisRaft:
    quorum: 3
EOF

echo "Validator $PEER_INDEX initialization complete"
echo "Logical validator clients assigned to this peer: $LOGICAL_VALIDATORS_PER_PEER"
echo "Starting peer..."

# Note: In production, peer would be started by systemd or docker
# This is just for initialization verification
peer version || true

echo "Setup complete for validator $PEER_INDEX"

#!/bin/bash
# AgentEcon Orderer Node Initialization Script

set -e

PROJECT_NAME=${project_name}

echo "Initializing AgentEcon Orderer..."

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

# Create orderer user
useradd -m -s /bin/bash orderer || true
usermod -aG docker orderer

# Clone or update AgentEcon repository
mkdir -p $GOPATH/src/github.com/agentecon
if [ ! -d "$GOPATH/src/github.com/agentecon/AgentEcon" ]; then
    git clone https://github.com/anonymous/agentecon.git $GOPATH/src/github.com/agentecon/AgentEcon
fi

cd $GOPATH/src/github.com/agentecon/AgentEcon
git pull origin main

# Setup Fabric orderer configuration
export FABRIC_CFG_PATH=/opt/fabric/config

mkdir -p /opt/fabric/config

cat > /opt/fabric/config/orderer.yaml <<EOF
General:
  ListenAddress: 0.0.0.0
  ListenPort: 7050
  TLS:
    Enabled: true
    PrivateKey: /etc/hyperledger/fabric/orderer/tls/server.key
    Certificate: /etc/hyperledger/fabric/orderer/tls/server.crt
    RootCAs:
      - /etc/hyperledger/fabric/orderer/tls/ca.crt
  GenesisMethod: file
  GenesisProfile: Initial
  GenesisFile: /etc/hyperledger/fabric/genesis.block
  LocalMSPDir: /etc/hyperledger/fabric/orderer/msp
  LocalMSPID: OrdererMSP
  Profile:
    Enabled: true
    Address: 0.0.0.0:6060
  Cluster:
    ClientCertificate: /etc/hyperledger/fabric/orderer/tls/server.crt
    ClientPrivateKey: /etc/hyperledger/fabric/orderer/tls/server.key
  Broadcast:
    AckDefaultConfig: 2
  Deliver:
    AckDefaultConfig: 2

FileLedger:
  Location: /var/hyperledger/orderer
  Prefix: hyperledgerch

Kafka:
  Retry:
    ShortInterval: 5s
    ShortTotal: 10s
    LongInterval: 5m
    LongTotal: 1h
    NetworkTimeout: 10s
  Producer:
    RetryBackoff: 100ms
  Consumer:
    RetryBackoff: 100ms

Debug:
  BroadcastTraceDir: /var/hyperledger/orderer/logs
  DeliverTraceDir: /var/hyperledger/orderer/logs

Operations:
  ListenAddress: 0.0.0.0:8443

Metrics:
  Provider: prometheus

Consensus:
  WALDir: /var/hyperledger/orderer/consensus
  SnapDir: /var/hyperledger/orderer/consensus
EOF

echo "Orderer initialization complete"
echo "Setup complete for orderer"

#!/bin/bash
# Simple deployment script for DigitalOcean

set -e

echo "🚀 Deploying Grid Trading Bot to Production..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running as trader user
if [ "$USER" != "trader" ]; then
    echo -e "${RED}❌ Please run as trader user: su - trader${NC}"
    exit 1
fi

# Check if .env.production exists
if [ ! -f .env.production ]; then
    echo -e "${RED}❌ .env.production file not found!${NC}"
    echo -e "${YELLOW}Please create .env.production with your API keys${NC}"
    exit 1
fi

# Stop existing containers
echo -e "${YELLOW}📦 Stopping existing containers...${NC}"
docker-compose -f docker-compose.production.yml down || true

# Build new image
echo -e "${YELLOW}🔨 Building trading bot image...${NC}"
docker-compose -f docker-compose.production.yml build

# Start services
echo -e "${YELLOW}🚀 Starting trading bot...${NC}"
docker-compose -f docker-compose.production.yml up -d

# Wait for services to be ready
echo -e "${YELLOW}⏳ Waiting for services to start...${NC}"
sleep 30

# Check if services are running
echo -e "${YELLOW}🔍 Checking service status...${NC}"
docker-compose -f docker-compose.production.yml ps

# Show logs
echo -e "${GREEN}✅ Deployment complete!${NC}"
echo -e "${GREEN}📊 Access Grafana dashboard: http://YOUR_SERVER_IP:3000${NC}"
echo -e "${GREEN}📝 View logs: docker-compose -f docker-compose.production.yml logs -f trading-bot${NC}"
echo -e "${GREEN}🛑 Stop bot: docker-compose -f docker-compose.production.yml down${NC}"

echo -e "\n${YELLOW}📋 Next Steps:${NC}"
echo "1. Monitor logs for first 30 minutes"
echo "2. Check Grafana dashboard at http://YOUR_SERVER_IP:3000"
echo "3. Verify trades on Binance"
echo "4. Set up alerts if needed"
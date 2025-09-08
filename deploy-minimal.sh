#!/bin/bash
# Minimal deployment script for low-memory VPS

set -e

echo "🚀 Deploying Minimal Grid Trading Bot..."

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
docker compose -f docker-compose.minimal.yml --env-file .env.production down || true
docker compose -f docker-compose.production.yml --env-file .env.production down || true

# Clean up to free memory
echo -e "${YELLOW}🧹 Cleaning up containers and images...${NC}"
docker container prune -f
docker image prune -f

# Show available memory
echo -e "${YELLOW}💾 Available memory:${NC}"
free -h

# Build new image
echo -e "${YELLOW}🔨 Building trading bot image...${NC}"
docker compose -f docker-compose.minimal.yml --env-file .env.production build

# Start minimal services
echo -e "${YELLOW}🚀 Starting minimal trading bot...${NC}"
docker compose -f docker-compose.minimal.yml --env-file .env.production up -d

# Wait for services to be ready
echo -e "${YELLOW}⏳ Waiting for services to start...${NC}"
sleep 30

# Check if services are running
echo -e "${YELLOW}🔍 Checking service status...${NC}"
docker compose -f docker-compose.minimal.yml --env-file .env.production ps

# Show logs
echo -e "${GREEN}✅ Minimal deployment complete!${NC}"
echo -e "${GREEN}📝 View logs: docker compose -f docker-compose.minimal.yml --env-file .env.production logs -f trading-bot${NC}"
echo -e "${GREEN}🛑 Stop bot: docker compose -f docker-compose.minimal.yml --env-file .env.production down${NC}"

echo -e "\n${YELLOW}📋 Next Steps:${NC}"
echo "1. Monitor logs for first 30 minutes"
echo "2. Check trades on Binance"
echo "3. Monitor system memory usage"
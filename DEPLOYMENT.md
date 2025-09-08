# Production Deployment Guide

## Quick Deployment to DigitalOcean

### Prerequisites
- DigitalOcean droplet with Ubuntu 22.04
- Binance API keys with SPOT trading permissions
- API key IP restrictions configured

### Files Overview
- `Dockerfile` - Production container configuration
- `docker-compose.production.yml` - Production services setup
- `deploy.sh` - Automated deployment script
- `config/config_live.json` - Live trading configuration
- `.env.production.template` - Environment variables template

### Deployment Steps

1. **Set up your environment variables:**
   ```bash
   cp .env.production.template .env.production
   # Edit .env.production with your real API keys
   nano .env.production
   ```

2. **Deploy to server:**
   ```bash
   chmod +x deploy.sh
   ./deploy.sh
   ```

3. **Monitor the bot:**
   ```bash
   # View logs
   docker-compose -f docker-compose.production.yml logs -f trading-bot
   
   # Check status
   docker-compose -f docker-compose.production.yml ps
   ```

### Safety Configuration
- Initial balance: $100 (start small!)
- Stop-loss enabled at $160
- Take-profit enabled at $220
- Grid range: $196-$236 (SOL/USDT)
- Only 5 grid levels for simplicity

### Monitoring
- Grafana dashboard: `http://YOUR_SERVER_IP:3000`
- Login: admin / [your_password_from_env]

### Emergency Commands
```bash
# Stop trading immediately
docker-compose -f docker-compose.production.yml down

# Restart bot
docker-compose -f docker-compose.production.yml restart trading-bot
```

## Security Notes
- Never commit .env.production to git
- Use IP restrictions on API keys
- Start with small amounts
- Monitor closely for first few hours
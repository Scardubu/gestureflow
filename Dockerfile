# Dockerfile - Main API Container

FROM python:3.9-slim

# Set working directory

WORKDIR /app

# Set environment variables

ENV PYTHONUNBUFFERED=1   
PYTHONDONTWRITEBYTECODE=1   
PIP_NO_CACHE_DIR=1   
PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies

RUN apt-get update && apt-get install -y   
build-essential   
curl   
&& rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching

COPY requirements.txt .
RUN pip install –no-cache-dir -r requirements.txt

# Copy application code

COPY . .

# Create necessary directories

RUN mkdir -p logs models/checkpoints data/dictionaries

# Expose API port

EXPOSE 8000

# Health check

HEALTHCHECK –interval=30s –timeout=10s –start-period=5s –retries=3   
CMD curl -f http://localhost:8000/health || exit 1

# Run API server

CMD [“uvicorn”, “api.main:app”, “–host”, “0.0.0.0”, “–port”, “8000”, “–workers”, “4”]

# ================================

# docker-compose.yml - Multi-container Setup

# ================================

# version: ‘3.8’

# 

# services:

# api:

# build:

# context: .

# dockerfile: Dockerfile

# container_name: gestureflow-api

# ports:

# - “8000:8000”

# volumes:

# - ./models:/app/models

# - ./data:/app/data

# - ./logs:/app/logs

# environment:

# - GESTUREFLOW_ENV=production

# - PYTHONPATH=/app

# restart: unless-stopped

# healthcheck:

# test: [“CMD”, “curl”, “-f”, “http://localhost:8000/health”]

# interval: 30s

# timeout: 10s

# retries: 3

# start_period: 40s

# 

# web:

# build:

# context: ./web

# dockerfile: Dockerfile

# container_name: gestureflow-web

# ports:

# - “3000:3000”

# environment:

# - NEXT_PUBLIC_API_URL=http://api:8000

# - NODE_ENV=production

# depends_on:

# - api

# restart: unless-stopped

# 

# networks:

# default:

# name: gestureflow-network

# ================================

# web/Dockerfile - Frontend Container

# ================================

# FROM node:18-alpine AS base

# 

# # Install dependencies only when needed

# FROM base AS deps

# RUN apk add –no-cache libc6-compat

# WORKDIR /app

# 

# COPY package.json package-lock.json* ./

# RUN npm ci

# 

# # Rebuild the source code only when needed

# FROM base AS builder

# WORKDIR /app

# COPY –from=deps /app/node_modules ./node_modules

# COPY . .

# 

# RUN npm run build

# 

# # Production image

# FROM base AS runner

# WORKDIR /app

# 

# ENV NODE_ENV=production

# 

# RUN addgroup –system –gid 1001 nodejs

# RUN adduser –system –uid 1001 nextjs

# 

# COPY –from=builder /app/public ./public

# COPY –from=builder –chown=nextjs:nodejs /app/.next/standalone ./

# COPY –from=builder –chown=nextjs:nodejs /app/.next/static ./.next/static

# 

# USER nextjs

# 

# EXPOSE 3000

# 

# ENV PORT=3000

# 

# CMD [“node”, “server.js”]

# ================================

# .dockerignore - Exclude from Docker Build

# ================================

# **pycache**

# *.pyc

# *.pyo

# *.pyd

# .Python

# venv/

# env/

# .venv/

# 

# # Node

# node_modules/

# npm-debug.log

# 

# # IDE

# .vscode/

# .idea/

# 

# # Git

# .git/

# .gitignore

# 

# # Data (too large)

# data/raw/*

# data/processed/*

# 

# # Models (mount as volume instead)

# models/checkpoints/*

# 

# # Logs

# logs/*

# *.log

# 

# # Tests

# tests/

# .pytest_cache/

# 

# # Documentation

# docs/

# *.md

# 

# # Other

# .DS_Store

# *.swp
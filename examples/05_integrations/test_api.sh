#!/bin/bash

# Test script for Mesh Agent Execution API
# Usage: ./test_api.sh

API_URL="http://localhost:8000"

echo "========================================="
echo "Testing Mesh Agent Execution API"
echo "========================================="
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test 1: Health check
echo -e "${YELLOW}Test 1: Health Check${NC}"
curl -s "${API_URL}/health" | python3 -m json.tool
echo ""
echo ""

# Test 2: List agents
echo -e "${YELLOW}Test 2: List Available Agents${NC}"
curl -s "${API_URL}/api/agents" | python3 -m json.tool
echo ""
echo ""

# Test 3: List tools
echo -e "${YELLOW}Test 3: List Available Tools${NC}"
curl -s "${API_URL}/api/tools" | python3 -m json.tool
echo ""
echo ""

# Test 4: Execute simple graph (sync)
echo -e "${YELLOW}Test 4: Execute Simple Graph (Synchronous)${NC}"
cat > /tmp/test_flow.json << 'EOF'
{
  "nodes": [
    {
      "id": "start_0",
      "type": "startAgentflow",
      "data": {
        "name": "startAgentflow",
        "label": "Start",
        "inputs": {}
      }
    },
    {
      "id": "llm_1",
      "type": "llmAgentflow",
      "data": {
        "name": "llmAgentflow",
        "label": "LLM",
        "inputs": {
          "model": "gpt-4o",
          "systemPrompt": "You are a helpful assistant. Be very concise."
        }
      }
    },
    {
      "id": "end_0",
      "type": "endAgentflow",
      "data": {
        "name": "endAgentflow",
        "label": "End",
        "inputs": {}
      }
    }
  ],
  "edges": [
    {"source": "start_0", "target": "llm_1"},
    {"source": "llm_1", "target": "end_0"}
  ]
}
EOF

curl -s -X POST "${API_URL}/api/execute-sync" \
  -H "Content-Type: application/json" \
  -d '{
    "flow": '"$(cat /tmp/test_flow.json)"',
    "input": "What is 2+2?",
    "session_id": "test-session-1"
  }' | python3 -m json.tool

echo ""
echo ""

# Test 5: Execute with streaming (sample - shows first few events)
echo -e "${YELLOW}Test 5: Execute with Streaming (First 10 Events)${NC}"
curl -s -X POST "${API_URL}/api/execute" \
  -H "Content-Type: application/json" \
  -d '{
    "flow": '"$(cat /tmp/test_flow.json)"',
    "input": "Tell me a very short joke about programming",
    "session_id": "test-session-2"
  }' | head -n 10

echo ""
echo ""

# Test 6: Execute agent graph (if agents available)
echo -e "${YELLOW}Test 6: Execute Agent Graph (if agents registered)${NC}"
if [ -f "examples/05_integrations/flows/agent_flow.json" ]; then
  curl -s -X POST "${API_URL}/api/execute-sync" \
    -H "Content-Type: application/json" \
    -d '{
      "flow": '"$(cat examples/05_integrations/flows/agent_flow.json)"',
      "input": "Explain quantum computing in simple terms",
      "session_id": "test-session-3"
    }' | python3 -m json.tool
else
  echo "⚠️  agent_flow.json not found, skipping"
fi

echo ""
echo ""

echo "========================================="
echo -e "${GREEN}Testing Complete!${NC}"
echo "========================================="
echo ""
echo "To test streaming in real-time, try:"
echo "  curl -X POST ${API_URL}/api/execute \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"flow\": ..., \"input\": \"...\"}'"
echo ""

/**
 * React Frontend Example for Mesh Agent Execution API
 *
 * This example demonstrates how to:
 * 1. Fetch available agents from backend
 * 2. Build React Flow graph with agent references
 * 3. Execute graph with streaming
 * 4. Display results in real-time
 *
 * Install dependencies:
 *   npm install @xyflow/react eventsource-parser
 */

import React, { useState, useEffect } from 'react';
import { ReactFlow, Node, Edge, useNodesState, useEdgesState } from '@xyflow/react';

// Type definitions
interface AgentInfo {
  id: string;
  name: string;
  type: 'vel' | 'openai';
  description?: string;
}

interface ToolInfo {
  id: string;
  name: string;
  description?: string;
}

interface ExecutionEvent {
  type: string;
  node_id?: string;
  content?: string;
  output?: any;
  error?: string;
  timestamp?: string;
  metadata?: Record<string, any>;
}

const API_BASE_URL = 'http://localhost:8000';

export function MeshGraphExecutor() {
  // State
  const [agents, setAgents] = useState<AgentInfo[]>([]);
  const [tools, setTools] = useState<ToolInfo[]>([]);
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [input, setInput] = useState('');
  const [executing, setExecuting] = useState(false);
  const [events, setEvents] = useState<ExecutionEvent[]>([]);
  const [streamedText, setStreamedText] = useState('');

  // Fetch available agents on mount
  useEffect(() => {
    fetchAgents();
    fetchTools();
  }, []);

  const fetchAgents = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/agents`);
      const data = await response.json();
      setAgents(data.agents);
    } catch (error) {
      console.error('Failed to fetch agents:', error);
    }
  };

  const fetchTools = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/tools`);
      const data = await response.json();
      setTools(data.tools);
    } catch (error) {
      console.error('Failed to fetch tools:', error);
    }
  };

  const addAgentNode = (agentId: string) => {
    const newNode: Node = {
      id: `agent_${Date.now()}`,
      type: 'agentAgentflow',
      position: { x: Math.random() * 500, y: Math.random() * 300 },
      data: {
        name: 'agentAgentflow',
        label: agents.find(a => a.id === agentId)?.name || agentId,
        inputs: {
          agent: agentId,  // 👈 String reference to backend agent!
          systemPrompt: 'Process: {{$question}}'
        }
      }
    };
    setNodes((nds) => [...nds, newNode]);
  };

  const addToolNode = (toolId: string) => {
    const newNode: Node = {
      id: `tool_${Date.now()}`,
      type: 'toolAgentflow',
      position: { x: Math.random() * 500, y: Math.random() * 300 },
      data: {
        name: 'toolAgentflow',
        label: tools.find(t => t.id === toolId)?.name || toolId,
        inputs: {
          tool: toolId  // 👈 String reference to backend tool!
        }
      }
    };
    setNodes((nds) => [...nds, newNode]);
  };

  const executeGraph = async () => {
    if (!input.trim() || nodes.length === 0) {
      alert('Please add nodes and provide input');
      return;
    }

    setExecuting(true);
    setEvents([]);
    setStreamedText('');

    try {
      // Build React Flow JSON from current nodes and edges
      const flowJson = {
        nodes: nodes.map(node => ({
          id: node.id,
          type: node.type,
          data: node.data,
          position: node.position
        })),
        edges: edges.map(edge => ({
          source: edge.source,
          target: edge.target,
          sourceHandle: edge.sourceHandle,
          targetHandle: edge.targetHandle
        }))
      };

      const response = await fetch(`${API_BASE_URL}/api/execute`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          flow: flowJson,
          input: input,
          session_id: `session-${Date.now()}`
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      // Parse SSE stream
      const reader = response.body?.getReader();
      const decoder = new TextDecoder();

      if (!reader) {
        throw new Error('No response body');
      }

      while (true) {
        const { done, value } = await reader.read();

        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const eventData = JSON.parse(line.slice(6));

            // Add to events log
            setEvents(prev => [...prev, eventData]);

            // Accumulate streamed text
            if (eventData.type === 'token' && eventData.content) {
              setStreamedText(prev => prev + eventData.content);
            }

            // Log execution complete
            if (eventData.type === 'execution_complete') {
              console.log('Execution complete:', eventData);
            }
          }
        }
      }
    } catch (error) {
      console.error('Execution failed:', error);
      alert(`Execution failed: ${error}`);
    } finally {
      setExecuting(false);
    }
  };

  return (
    <div style={{ display: 'flex', height: '100vh' }}>
      {/* Left sidebar: Agent/Tool selection */}
      <div style={{ width: '250px', padding: '20px', borderRight: '1px solid #ccc', overflowY: 'auto' }}>
        <h2>Available Agents</h2>
        {agents.map(agent => (
          <div key={agent.id} style={{ marginBottom: '10px' }}>
            <button
              onClick={() => addAgentNode(agent.id)}
              style={{ width: '100%', padding: '8px' }}
            >
              + {agent.name}
            </button>
            <small style={{ display: 'block', color: '#666' }}>
              {agent.type} • {agent.description}
            </small>
          </div>
        ))}

        <h2 style={{ marginTop: '30px' }}>Available Tools</h2>
        {tools.map(tool => (
          <div key={tool.id} style={{ marginBottom: '10px' }}>
            <button
              onClick={() => addToolNode(tool.id)}
              style={{ width: '100%', padding: '8px' }}
            >
              + {tool.name}
            </button>
            <small style={{ display: 'block', color: '#666' }}>
              {tool.description}
            </small>
          </div>
        ))}
      </div>

      {/* Center: React Flow graph editor */}
      <div style={{ flex: 1, position: 'relative' }}>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          fitView
        />

        {/* Execution controls */}
        <div style={{
          position: 'absolute',
          top: '20px',
          right: '20px',
          background: 'white',
          padding: '15px',
          borderRadius: '8px',
          boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
          minWidth: '300px'
        }}>
          <h3>Execute Graph</h3>
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Enter your question..."
            style={{
              width: '100%',
              padding: '8px',
              marginBottom: '10px',
              minHeight: '60px'
            }}
            disabled={executing}
          />
          <button
            onClick={executeGraph}
            disabled={executing}
            style={{
              width: '100%',
              padding: '10px',
              background: executing ? '#ccc' : '#0066cc',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: executing ? 'not-allowed' : 'pointer'
            }}
          >
            {executing ? 'Executing...' : 'Execute'}
          </button>
        </div>
      </div>

      {/* Right sidebar: Execution results */}
      <div style={{
        width: '350px',
        padding: '20px',
        borderLeft: '1px solid #ccc',
        overflowY: 'auto',
        background: '#f9f9f9'
      }}>
        <h2>Execution Results</h2>

        {streamedText && (
          <div style={{
            background: 'white',
            padding: '15px',
            borderRadius: '4px',
            marginBottom: '20px',
            whiteSpace: 'pre-wrap'
          }}>
            {streamedText}
          </div>
        )}

        <h3>Event Log</h3>
        <div style={{ fontSize: '12px' }}>
          {events.map((event, idx) => (
            <div key={idx} style={{
              padding: '8px',
              marginBottom: '5px',
              background: event.type === 'execution_error' ? '#fee' : 'white',
              borderRadius: '4px',
              borderLeft: `3px solid ${
                event.type === 'node_start' ? '#0066cc' :
                event.type === 'node_complete' ? '#00cc66' :
                event.type === 'execution_error' ? '#cc0000' :
                '#ccc'
              }`
            }}>
              <div style={{ fontWeight: 'bold' }}>{event.type}</div>
              {event.node_id && <div>Node: {event.node_id}</div>}
              {event.error && <div style={{ color: 'red' }}>{event.error}</div>}
              {event.metadata && (
                <div style={{ fontSize: '10px', color: '#666' }}>
                  {JSON.stringify(event.metadata, null, 2)}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default MeshGraphExecutor;

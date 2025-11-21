---
layout: default
title: Concepts
nav_order: 4
has_children: true
---

# Core Concepts

Understanding the fundamental concepts behind Mesh.

## Overview

Mesh is built around several core concepts that work together to create powerful multi-agent workflows:

1. **[Graphs](concepts/graphs)** - The structure of your workflow
2. **[Nodes](concepts/nodes)** - The building blocks of computation
3. **[Execution](concepts/execution)** - How workflows run
4. **[Events](concepts/events)** - Real-time streaming feedback
5. **[Variables](concepts/variables)** - Dynamic data flow and templating

## Quick Summary

### Graphs
Directed graphs with controlled cycles that define workflow structure and execution flow.

### Nodes
Seven types of nodes that perform specific tasks: Start, End, Agent, LLM, Tool, Condition, and Loop.

### Execution
Queue-based execution model with dependency tracking and state management.

### Events
Provider-agnostic streaming events for real-time feedback during execution.

### Variables
Template-based system for referencing data across nodes using `{{variable}}` syntax, with automatic natural language parsing.

## Next Steps

Dive into each concept:
- [Graphs](concepts/graphs) - Learn about graph structure
- [Nodes](concepts/nodes) - Understand all 7 node types
- [Execution](concepts/execution) - See how workflows execute
- [Events](concepts/events) - Handle streaming events
- [Variables](concepts/variables) - Use dynamic templates and data flow

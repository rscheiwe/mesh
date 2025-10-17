"""Orchestration wrapper for Vel event translators.

This module provides a helper to fill the event gaps when using Vel's translators
directly. Vel translators only emit content-level events (text-delta, tool-input-*),
but not orchestration events (start, start-step, finish-step, finish).

Based on: https://rscheiwe.github.io/vel/event-translators/using-translators

This helper:
1. Emits missing orchestration events (start, start-step, finish-step, finish)
2. Tracks internal metadata (response-metadata, finish-message)
3. Detects step boundaries in multi-step tool execution
4. Ensures AI SDK frontend compatibility
"""

from typing import Any, Dict, Optional, AsyncIterator
from dataclasses import dataclass, field


@dataclass
class StepMetadata:
    """Tracks metadata for a single step."""

    step_index: int = 0
    response_id: Optional[str] = None
    model_id: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None
    finish_reason: str = "stop"
    has_tool_calls: bool = False
    tool_calls_completed: int = 0


class TranslatorOrchestrator:
    """Orchestrates Vel translator events with proper start/finish boundaries.

    This class wraps a Vel SDK translator and fills the event gaps by:
    - Emitting orchestration events (start, start-step, finish-step, finish)
    - Tracking internal metadata from response-metadata and finish-message
    - Detecting step boundaries when tools are used
    - Accumulating total usage across steps

    Usage:
        >>> orchestrator = TranslatorOrchestrator(translator)
        >>> async for event in orchestrator.stream(native_events):
        ...     # Events now include start/finish/step boundaries
        ...     yield event
    """

    def __init__(self, translator: Any, max_steps: int = 10):
        """Initialize orchestrator.

        Args:
            translator: Vel SDK translator instance (e.g., OpenAIAgentsSDKTranslator)
            max_steps: Maximum number of steps to allow (safety limit)
        """
        self.translator = translator
        self.max_steps = max_steps
        self._reset_state()

    def _reset_state(self):
        """Reset orchestrator state for new execution."""
        self.current_step = StepMetadata()
        self.total_usage: Dict[str, int] = {}
        self.steps_completed = 0
        self.execution_started = False
        self.current_step_started = False
        self.waiting_for_new_step = False

    async def stream(
        self,
        native_event_stream: AsyncIterator[Any],
        emit_start: bool = True,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream events with orchestration boundaries.

        This is the main method that wraps the translator and emits complete
        event stream including start/finish/step events.

        Args:
            native_event_stream: Stream of native events from agent SDK
            emit_start: Whether to emit the 'start' event (set False if already emitted)

        Yields:
            Dict[str, Any]: Complete event stream with orchestration events
        """
        self._reset_state()

        # 1. Emit start event
        if emit_start and not self.execution_started:
            yield {'type': 'start'}
            self.execution_started = True

        # 2. Emit initial start-step
        if not self.current_step_started:
            yield {'type': 'start-step', 'stepIndex': self.steps_completed}
            self.current_step_started = True

        # 3. Stream translated events
        async for native_event in native_event_stream:
            # Translate native event to Vel format
            vel_event = self.translator.translate(native_event)

            if not vel_event:
                # Event was skipped (e.g., agent_updated_stream_event)
                continue

            event_dict = vel_event.to_dict()
            event_type = event_dict.get('type', '')

            # Handle internal metadata events (consume, don't forward)
            if event_type == 'response-metadata':
                self.current_step.response_id = event_dict.get('id')
                self.current_step.model_id = event_dict.get('modelId')
                self.current_step.usage = event_dict.get('usage')
                continue  # Don't forward

            elif event_type == 'finish-message':
                self.current_step.finish_reason = event_dict.get('finishReason', 'stop')
                continue  # Don't forward

            # Track tool calls
            elif event_type == 'tool-input-available':
                self.current_step.has_tool_calls = True
                yield event_dict

            elif event_type == 'tool-output-available':
                self.current_step.tool_calls_completed += 1
                self.waiting_for_new_step = True  # Next text-delta = new step
                yield event_dict

            # Detect step boundary: after tool execution, first text indicates new step
            elif event_type in ('text-start', 'text-delta') and self.waiting_for_new_step:
                # Finish previous step
                yield self._build_finish_step_event()

                # Start new step
                self.steps_completed += 1
                self.current_step = StepMetadata(step_index=self.steps_completed)
                self.current_step_started = True
                self.waiting_for_new_step = False

                yield {'type': 'start-step', 'stepIndex': self.steps_completed}

                # Now yield the text event that triggered boundary
                yield event_dict

            else:
                # Forward content events
                yield event_dict

        # 4. Finalize any pending tool calls
        if hasattr(self.translator, 'finalize_tool_calls'):
            for pending_event in self.translator.finalize_tool_calls():
                yield pending_event.to_dict()

        # 5. Emit finish-step for final step
        if self.current_step_started:
            yield self._build_finish_step_event()

        # 6. Emit finish event
        yield {
            'type': 'finish',
            'finishReason': self.current_step.finish_reason,
            'totalUsage': self.total_usage if self.total_usage else None,
            'stepsCompleted': self.steps_completed + 1,
        }

    def _build_finish_step_event(self) -> Dict[str, Any]:
        """Build finish-step event with metadata from current step.

        Returns:
            Dict with finish-step event data
        """
        # Accumulate usage
        if self.current_step.usage:
            for key, value in self.current_step.usage.items():
                self.total_usage[key] = self.total_usage.get(key, 0) + value

        return {
            'type': 'finish-step',
            'stepIndex': self.current_step.step_index,
            'finishReason': self.current_step.finish_reason,
            'usage': self.current_step.usage,
            'response': {
                'id': self.current_step.response_id,
                'modelId': self.current_step.model_id,
            } if self.current_step.response_id else None,
            'hadToolCalls': self.current_step.has_tool_calls,
        }


class SimpleTranslatorOrchestrator:
    """Simplified orchestrator for single-step, no-tool scenarios.

    This is a lighter-weight version that doesn't do step boundary detection,
    suitable for simple LLM calls without tool execution.

    Usage:
        >>> orchestrator = SimpleTranslatorOrchestrator(translator)
        >>> async for event in orchestrator.stream(chunk_stream):
        ...     yield event
    """

    def __init__(self, translator: Any):
        """Initialize simple orchestrator.

        Args:
            translator: Vel API translator instance (e.g., OpenAIAPITranslator)
        """
        self.translator = translator
        self._reset_state()

    def _reset_state(self):
        """Reset state for new execution."""
        self.response_id: Optional[str] = None
        self.model_id: Optional[str] = None
        self.usage: Optional[Dict[str, Any]] = None
        self.finish_reason: str = 'stop'

    async def stream(
        self,
        chunk_stream: AsyncIterator[Any],
        emit_start: bool = True,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream events with start/finish boundaries.

        Args:
            chunk_stream: Stream of chunks from provider API
            emit_start: Whether to emit 'start' event

        Yields:
            Dict[str, Any]: Events with orchestration boundaries
        """
        self._reset_state()

        # 1. Emit start
        if emit_start:
            yield {'type': 'start'}

        # 2. Emit start-step
        yield {'type': 'start-step'}

        # 3. Stream translated events
        async for chunk in chunk_stream:
            event = self.translator.translate_chunk(chunk)

            if not event:
                continue

            event_dict = event.to_dict()
            event_type = event_dict.get('type', '')

            # Consume metadata
            if event_type == 'response-metadata':
                self.response_id = event_dict.get('id')
                self.model_id = event_dict.get('modelId')
                self.usage = event_dict.get('usage')
                continue

            elif event_type == 'finish-message':
                self.finish_reason = event_dict.get('finishReason', 'stop')
                continue

            # Forward content events
            yield event_dict

        # 4. Finalize pending tool calls if needed
        if hasattr(self.translator, 'finalize_tool_calls'):
            for pending in self.translator.finalize_tool_calls():
                yield pending.to_dict()

        # 5. Emit finish-step
        yield {
            'type': 'finish-step',
            'finishReason': self.finish_reason,
            'usage': self.usage,
            'response': {
                'id': self.response_id,
                'modelId': self.model_id,
            } if self.response_id else None,
        }

        # 6. Emit finish
        yield {
            'type': 'finish',
            'finishReason': self.finish_reason,
            'totalUsage': self.usage,
        }

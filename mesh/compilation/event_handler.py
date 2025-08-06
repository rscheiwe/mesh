"""Event handler for graph execution."""

import asyncio
from typing import Any, Callable, List, Optional

from mesh.core.events import Event


class EventHandler:
    """Handles event emissions during graph execution.
    
    This class decouples event handling from execution mode (streaming vs batch),
    allowing both GraphExecutor and StreamingGraphExecutor to emit the same events.
    """
    
    def __init__(self):
        """Initialize the event handler."""
        self._listeners: List[Callable[[Event], Any]] = []
    
    def add_listener(self, listener: Callable[[Event], Any]) -> None:
        """Add an event listener.
        
        Args:
            listener: Callable that receives Event objects.
                     Can be sync or async.
        """
        self._listeners.append(listener)
    
    def remove_listener(self, listener: Callable[[Event], Any]) -> None:
        """Remove an event listener.
        
        Args:
            listener: The listener to remove.
        """
        if listener in self._listeners:
            self._listeners.remove(listener)
    
    async def emit(self, event: Event) -> None:
        """Emit an event to all listeners.
        
        Args:
            event: The event to emit.
        """
        for listener in self._listeners:
            try:
                if asyncio.iscoroutinefunction(listener):
                    await listener(event)
                else:
                    # Run sync listeners in thread pool to avoid blocking
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, listener, event)
            except Exception as e:
                # Log error but don't fail execution
                # In production, this should use proper logging
                print(f"Error in event listener: {e}")
    
    def clear_listeners(self) -> None:
        """Remove all event listeners."""
        self._listeners.clear()


class EventCollector:
    """Collects events for later analysis.
    
    This is a utility class that can be used as an event listener
    to collect all events during execution.
    """
    
    def __init__(self):
        """Initialize the event collector."""
        self.events: List[Event] = []
    
    def __call__(self, event: Event) -> None:
        """Collect an event.
        
        Args:
            event: The event to collect.
        """
        self.events.append(event)
    
    def get_events_by_type(self, event_type: str) -> List[Event]:
        """Get all events of a specific type.
        
        Args:
            event_type: The type of events to retrieve.
            
        Returns:
            List of events matching the type.
        """
        return [e for e in self.events if e.type == event_type]
    
    def get_node_events(self, node_id: str) -> List[Event]:
        """Get all events for a specific node.
        
        Args:
            node_id: The ID of the node.
            
        Returns:
            List of events for the node.
        """
        return [e for e in self.events if getattr(e, 'node_id', None) == node_id]
    
    def clear(self) -> None:
        """Clear all collected events."""
        self.events.clear()
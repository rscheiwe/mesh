"""Utility functions for working with events."""

from typing import Optional

from mesh.core.events import Event, EventType


async def print_event(event: Event, prefix: str = "  ") -> None:
    """Print an event in a human-readable format.
    
    This is useful for debugging and examples to visualize event flow.
    
    Args:
        event: The event to print
        prefix: Optional prefix for indentation (default: "  ")
    """
    if event.type == EventType.GRAPH_START:
        print(f"{prefix}📊 Graph execution started")
    elif event.type == EventType.GRAPH_END:
        execution_time = event.data.get('execution_time', 0)
        success = event.data.get('success', False)
        status = "✅" if success else "❌"
        print(f"{prefix}{status} Graph execution completed in {execution_time:.3f}s")
    elif event.type == EventType.NODE_START:
        print(f"{prefix}🟢 Node starting: {event.node_name}")
    elif event.type == EventType.NODE_END:
        execution_time = event.data.get('execution_time', 0)
        success = event.data.get('success', False)
        status = "✅" if success else "⚠️"
        print(f"{prefix}{status} Node completed: {event.node_name} ({execution_time:.3f}s)")
    elif event.type == EventType.NODE_ERROR:
        error = event.data.get('error', 'Unknown error')
        print(f"{prefix}❌ Node error in {event.node_name}: {error}")
    elif event.type == EventType.GRAPH_ERROR:
        error = event.data.get('error', 'Unknown error')
        print(f"{prefix}❌ Graph error: {error}")
    elif event.type == EventType.TOOL_START:
        tool_name = event.data.get('tool_name', 'unknown')
        print(f"{prefix}🔧 Tool starting: {tool_name}")
    elif event.type == EventType.TOOL_END:
        tool_name = event.data.get('tool_name', 'unknown')
        success = event.data.get('success', False)
        status = "✅" if success else "❌"
        print(f"{prefix}{status} Tool completed: {tool_name}")
    elif event.type == EventType.TOOL_ERROR:
        tool_name = event.data.get('tool_name', 'unknown')
        error = event.data.get('error', 'Unknown error')
        print(f"{prefix}❌ Tool error in {tool_name}: {error}")
    elif event.type == EventType.STREAM_CHUNK:
        content = event.data.get('content', '')
        # For stream chunks, print inline without newline
        print(content, end='', flush=True)
    elif event.type == EventType.STATE_UPDATE:
        key = event.data.get('key', 'unknown')
        value = event.data.get('value', None)
        print(f"{prefix}📝 State updated: {key} = {value}")
    else:
        # Generic fallback for custom events
        print(f"{prefix}📌 {event.type}: {event.data}")


def create_print_listener(prefix: str = "  "):
    """Create an event listener function that prints events.
    
    Args:
        prefix: Optional prefix for indentation (default: "  ")
        
    Returns:
        An async function that can be used as an event listener
        
    Example:
        event_handler = EventHandler()
        event_handler.add_listener(create_print_listener(">> "))
    """
    async def listener(event: Event):
        await print_event(event, prefix)
    return listener


class EventPrinter:
    """A simple event listener class that prints events.
    
    This can be used directly as a listener or subclassed for custom behavior.
    
    Example:
        event_handler = EventHandler()
        printer = EventPrinter(prefix="  ")
        event_handler.add_listener(printer)
    """
    
    def __init__(self, prefix: str = "  ", verbose: bool = True):
        """Initialize the event printer.
        
        Args:
            prefix: Prefix for indentation
            verbose: If True, print all events. If False, only print errors.
        """
        self.prefix = prefix
        self.verbose = verbose
    
    async def __call__(self, event: Event) -> None:
        """Handle an event by printing it.
        
        Args:
            event: The event to handle
        """
        # Skip non-error events if not verbose
        if not self.verbose:
            if event.type not in [EventType.NODE_ERROR, EventType.GRAPH_ERROR, 
                                 EventType.TOOL_ERROR]:
                return
        
        await print_event(event, self.prefix)


def format_event_summary(events: list[Event]) -> str:
    """Format a summary of events.
    
    Args:
        events: List of events to summarize
        
    Returns:
        A formatted string summary
    """
    if not events:
        return "No events recorded"
    
    # Count event types
    event_counts = {}
    total_execution_time = 0
    errors = []
    
    for event in events:
        event_type = event.type
        event_counts[event_type] = event_counts.get(event_type, 0) + 1
        
        # Track errors
        if event.type in [EventType.NODE_ERROR, EventType.GRAPH_ERROR, EventType.TOOL_ERROR]:
            errors.append(event)
        
        # Track execution time
        if event.type == EventType.GRAPH_END:
            total_execution_time = event.data.get('execution_time', 0)
    
    # Build summary
    lines = ["Event Summary:", "=" * 40]
    
    # Event counts
    lines.append("\nEvent Counts:")
    for event_type, count in sorted(event_counts.items()):
        lines.append(f"  {event_type}: {count}")
    
    # Execution time
    if total_execution_time > 0:
        lines.append(f"\nTotal Execution Time: {total_execution_time:.3f}s")
    
    # Errors
    if errors:
        lines.append(f"\nErrors ({len(errors)}):")
        for error in errors:
            error_msg = error.data.get('error', 'Unknown error')
            if error.node_name:
                lines.append(f"  - {error.node_name}: {error_msg}")
            else:
                lines.append(f"  - {error_msg}")
    else:
        lines.append("\nNo errors occurred")
    
    return "\n".join(lines)
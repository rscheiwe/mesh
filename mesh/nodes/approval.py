"""Approval node for human-in-the-loop workflows.

This node pauses execution until an external approval signal is received.
Used for plan review, content moderation, or any workflow requiring human oversight.
"""

from typing import Any, Dict, Optional

from mesh.nodes.base import BaseNode, NodeResult
from mesh.core.state import ExecutionContext
from mesh.core.events import ExecutionEvent, EventType


class ApprovalNode(BaseNode):
    """Human-in-the-loop approval node that pauses execution for external approval.

    When executed, this node:
    1. Emits an APPROVAL_PENDING event with approval data
    2. Returns a NodeResult with approval_pending=True
    3. The executor detects this and pauses execution
    4. External system (API/webhook) calls executor.resume() with approval result
    5. Execution continues from the next node (if approved) or ends (if rejected)

    The approval_data in the output can contain any information needed for
    the approval decision, such as a generated plan to review.

    Args:
        id: Node identifier
        approval_id: Unique identifier for this approval point (for tracking)
        approval_message: Human-readable message explaining what needs approval
        timeout_seconds: Optional timeout in seconds (None = wait forever)
        data_extractor: Optional function to extract approval data from input
        event_mode: Event emission mode (default: "full")
        config: Additional configuration

    Example:
        >>> approval_node = ApprovalNode(
        ...     id="plan_approval",
        ...     approval_id="research_plan_v1",
        ...     approval_message="Please review the research plan before execution",
        ... )

    Example with data extraction:
        >>> def extract_plan(input):
        ...     return {"plan": input.get("plan"), "steps": input.get("steps", [])}
        >>>
        >>> approval_node = ApprovalNode(
        ...     id="plan_approval",
        ...     approval_id="research_plan",
        ...     approval_message="Review research plan",
        ...     data_extractor=extract_plan,
        ... )
    """

    def __init__(
        self,
        id: str,
        approval_id: str,
        approval_message: str = "Approval required to continue",
        timeout_seconds: Optional[int] = None,
        data_extractor: Optional[callable] = None,
        event_mode: str = "full",
        config: Dict[str, Any] = None,
    ):
        """Initialize approval node.

        Args:
            id: Node identifier
            approval_id: Unique identifier for this approval request
            approval_message: Message to display to approver
            timeout_seconds: Timeout in seconds (None = no timeout)
            data_extractor: Function to extract approval data from input
            event_mode: Event emission mode
            config: Additional configuration
        """
        super().__init__(id, config or {})
        self.approval_id = approval_id
        self.approval_message = approval_message
        self.timeout_seconds = timeout_seconds
        self.data_extractor = data_extractor
        self.event_mode = event_mode

    async def _execute_impl(
        self,
        input: Any,
        context: ExecutionContext,
    ) -> NodeResult:
        """Execute approval node - emit event and signal pause.

        Args:
            input: Input data (passed through and available to approver)
            context: Execution context

        Returns:
            NodeResult with approval_pending=True
        """
        # Extract approval data from input
        if self.data_extractor is not None:
            try:
                approval_data = self.data_extractor(input)
            except Exception as e:
                approval_data = {
                    "input": input,
                    "extraction_error": str(e),
                }
        else:
            # Default: pass through input as approval data
            if isinstance(input, dict):
                approval_data = input
            else:
                approval_data = {"input": input}

        # Add context information
        approval_data["approval_id"] = self.approval_id
        approval_data["approval_message"] = self.approval_message
        approval_data["session_id"] = context.session_id
        approval_data["graph_id"] = context.graph_id

        if self.timeout_seconds is not None:
            approval_data["timeout_seconds"] = self.timeout_seconds

        # Emit APPROVAL_PENDING event
        if self.event_mode != "silent":
            await context.emit_event(
                ExecutionEvent(
                    type=EventType.APPROVAL_PENDING,
                    node_id=self.id,
                    output=approval_data,
                    metadata={
                        "approval_id": self.approval_id,
                        "approval_message": self.approval_message,
                        "timeout_seconds": self.timeout_seconds,
                        "node_type": "approval",
                    },
                )
            )

        # Return result signaling executor to pause
        return NodeResult(
            output=input,  # Pass through input for downstream nodes
            approval_pending=True,
            approval_id=self.approval_id,
            approval_data=approval_data,
            state={
                "_approval_pending": True,
                "_approval_id": self.approval_id,
                "_approval_node_id": self.id,
            },
            metadata={
                "approval_id": self.approval_id,
                "approval_message": self.approval_message,
                "node_type": "approval",
            },
        )

    def __repr__(self) -> str:
        return f"ApprovalNode(id='{self.id}', approval_id='{self.approval_id}')"


class ApprovalResult:
    """Result of an approval decision.

    This is passed to executor.resume() to continue execution
    after an approval node has paused.

    Attributes:
        approved: Whether the approval was granted
        modified_data: Optional modified data (e.g., edited plan)
        rejection_reason: Optional reason if rejected
        approver_id: Optional identifier of who approved
        metadata: Additional metadata about the decision
    """

    def __init__(
        self,
        approved: bool,
        modified_data: Optional[Dict[str, Any]] = None,
        rejection_reason: Optional[str] = None,
        approver_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.approved = approved
        self.modified_data = modified_data
        self.rejection_reason = rejection_reason
        self.approver_id = approver_id
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "approved": self.approved,
            "modified_data": self.modified_data,
            "rejection_reason": self.rejection_reason,
            "approver_id": self.approver_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ApprovalResult":
        """Create from dictionary."""
        return cls(
            approved=data.get("approved", False),
            modified_data=data.get("modified_data"),
            rejection_reason=data.get("rejection_reason"),
            approver_id=data.get("approver_id"),
            metadata=data.get("metadata", {}),
        )


# Convenience functions for creating approval results
def approve(
    modified_data: Optional[Dict[str, Any]] = None,
    approver_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> ApprovalResult:
    """Create an approval result indicating approval granted.

    Args:
        modified_data: Optional modified data to use instead of original
        approver_id: Optional identifier of who approved
        metadata: Additional metadata

    Returns:
        ApprovalResult with approved=True
    """
    return ApprovalResult(
        approved=True,
        modified_data=modified_data,
        approver_id=approver_id,
        metadata=metadata,
    )


def reject(
    reason: str = "Approval denied",
    approver_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> ApprovalResult:
    """Create an approval result indicating rejection.

    Args:
        reason: Reason for rejection
        approver_id: Optional identifier of who rejected
        metadata: Additional metadata

    Returns:
        ApprovalResult with approved=False
    """
    return ApprovalResult(
        approved=False,
        rejection_reason=reason,
        approver_id=approver_id,
        metadata=metadata,
    )

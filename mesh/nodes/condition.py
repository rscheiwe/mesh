"""Condition node for conditional branching.

This node evaluates conditions and determines which branch(es) to execute.
It's used for routing workflow based on data or state.
"""

from typing import List, Dict, Callable, Any, Optional, Union
from dataclasses import dataclass
import inspect

from mesh.nodes.base import BaseNode, NodeResult
from mesh.core.state import ExecutionContext
from mesh.core.events import ExecutionEvent, EventType, transform_event_for_transient_mode


# Type alias for predicates that can take input only or input + context
PredicateFunc = Union[Callable[[Any], bool], Callable[[Any, ExecutionContext], bool]]


@dataclass
class Condition:
    """Single branch condition.

    Attributes:
        name: Condition name/identifier
        predicate: Function that evaluates to True/False.
            Can be either:
            - Callable[[Any], bool] - takes only input
            - Callable[[Any, ExecutionContext], bool] - takes input and context
        target_node: Node ID to route to if condition is True

    Example (input only):
        >>> Condition(
        ...     name="is_valid",
        ...     predicate=lambda input: input.get("valid", False),
        ...     target_node="valid_handler"
        ... )

    Example (with context access):
        >>> def has_incomplete_steps(input, context):
        ...     plan = context.state.get("plan", {})
        ...     return any(not s.get("completed") for s in plan.get("steps", []))
        >>>
        >>> Condition(
        ...     name="has_more_steps",
        ...     predicate=has_incomplete_steps,
        ...     target_node="researcher"
        ... )
    """

    name: str
    predicate: PredicateFunc
    target_node: str


class ConditionNode(BaseNode):
    """Unified conditional branching node supporting both deterministic and AI-driven routing.

    This node can operate in two modes:
    1. **Deterministic** (default): Uses predicate functions for rule-based routing
    2. **AI**: Uses an LLM to analyze input and classify into scenarios

    The routing mode is controlled by the `condition_routing` parameter.

    ## Deterministic Mode (condition_routing="deterministic")

    Evaluates explicit conditions using logical operators. Fast, predictable, and
    ideal for clear rules.

    Supported Operations (via SimpleCondition):
        - equals: Check if value equals target
        - not_equal: Check if value does not equal target
        - contains: Check if string contains substring
        - not_contains: Check if string does not contain substring
        - greater_than: Check if number > threshold
        - less_than: Check if number < threshold
        - is_empty: Check if value is empty (None, "", [], {})

    ## AI Mode (condition_routing="ai")

    Uses an LLM to classify input into scenarios based on natural language instructions.
    Flexible, handles nuanced cases, ideal for intent recognition or complex routing.

    Args:
        id: Node identifier
        condition_routing: Routing mode - "deterministic" (default) or "ai"
        conditions: List of Condition objects (deterministic mode)
        model: LLM model to use (ai mode) - e.g., "gpt-4", "claude-3-5-sonnet-20241022"
        instructions: Natural language task description (ai mode)
        scenarios: List of scenario dicts with name, description, target (ai mode)
        default_target: Default node to route to if no conditions/scenarios match
        config: Additional configuration

    Example (Deterministic - Simple True/False):
        >>> from mesh.nodes import ConditionNode, SimpleCondition
        >>>
        >>> condition_node = ConditionNode(
        ...     id="age_check",
        ...     condition_routing="deterministic",
        ...     conditions=[
        ...         SimpleCondition.greater_than("age", 17, "adult_path"),
        ...         SimpleCondition.less_than("age", 18, "minor_path"),
        ...     ],
        ... )

    Example (Deterministic - Multiple Operations):
        >>> condition_node = ConditionNode(
        ...     id="request_router",
        ...     condition_routing="deterministic",
        ...     conditions=[
        ...         SimpleCondition.equals("status", "success", "success_handler"),
        ...         SimpleCondition.contains("message", "error", "error_handler"),
        ...     ],
        ...     default_target="fallback_handler",
        ... )

    Example (AI - Intent Classification):
        >>> condition_node = ConditionNode(
        ...     id="intent_router",
        ...     condition_routing="ai",
        ...     model="gpt-4",
        ...     instructions="Classify the user's request into sales, support, or billing",
        ...     scenarios=[
        ...         {"name": "sales", "description": "Questions about products or purchasing", "target": "sales_handler"},
        ...         {"name": "support", "description": "Technical issues or help", "target": "support_handler"},
        ...         {"name": "billing", "description": "Payment or invoice questions", "target": "billing_handler"},
        ...     ],
        ...     default_target="general_handler",
        ... )

    See Also:
        - examples/03_advanced_patterns/condition_node_routing.py
        - SimpleCondition: Helper class for common comparison operations
        - Condition: Dataclass for custom predicate functions
    """

    def __init__(
        self,
        id: str,
        condition_routing: str = "deterministic",
        conditions: Optional[List[Condition]] = None,
        model: Optional[str] = None,
        instructions: Optional[str] = None,
        scenarios: Optional[List[Dict[str, str]]] = None,
        default_target: Optional[str] = None,
        event_mode: str = "full",
        config: Dict[str, Any] = None,
    ):
        """Initialize condition node.

        Args:
            id: Node identifier
            condition_routing: "deterministic" or "ai"
            conditions: List of Condition objects (deterministic mode)
            model: LLM model (ai mode)
            instructions: Task description (ai mode)
            scenarios: List of scenario dicts (ai mode)
            default_target: Default node to route to if no match
            event_mode: Event emission mode (default: "full")
                - "full": All events - streams to chat
                - "status_only": Only progress indicators
                - "transient_events": All events prefixed with data-condition-node-*
                - "silent": No events
            config: Additional configuration
        """
        super().__init__(id, config or {})

        # Validate routing mode
        if condition_routing not in ["deterministic", "ai"]:
            raise ValueError(
                f"condition_routing must be 'deterministic' or 'ai', got: {condition_routing}"
            )

        self.condition_routing = condition_routing
        self.event_mode = event_mode
        self.default_target = default_target

        # Deterministic mode parameters
        if condition_routing == "deterministic":
            if conditions is None:
                raise ValueError("Deterministic mode requires 'conditions' parameter")
            self.conditions = conditions

        # AI mode parameters
        elif condition_routing == "ai":
            if model is None or instructions is None or scenarios is None:
                raise ValueError(
                    "AI mode requires 'model', 'instructions', and 'scenarios' parameters"
                )
            self.model = model
            self.instructions = instructions
            self.scenarios = scenarios

            # Validate scenarios
            for scenario in scenarios:
                if "name" not in scenario or "target" not in scenario:
                    raise ValueError(
                        f"Each scenario must have 'name' and 'target' keys, got: {scenario}"
                    )

    async def _emit_event_if_enabled(self, context: ExecutionContext, event: "ExecutionEvent") -> None:
        """Emit event based on event_mode.

        Args:
            context: Execution context
            event: Event to emit
        """
        # Silent mode - no events
        if self.event_mode == "silent":
            return

        # Status only - skip regular events, only custom events emitted separately
        if self.event_mode == "status_only":
            # Only emit if it's a custom data event
            if event.type == EventType.CUSTOM_DATA:
                await context.emit_event(event)
            return

        # Transient events - transform all events with data-condition-node-* prefix
        if self.event_mode == "transient_events":
            transformed_event = transform_event_for_transient_mode(event, "condition")
            await context.emit_event(transformed_event)
        else:
            # Full mode - emit events normally
            await context.emit_event(event)

    async def _execute_impl(
        self,
        input: Any,
        context: ExecutionContext,
    ) -> NodeResult:
        """Route execution based on condition_routing mode.

        Args:
            input: Input data to evaluate
            context: Execution context

        Returns:
            NodeResult with routing information in metadata
        """
        if self.condition_routing == "deterministic":
            return await self._execute_deterministic(input, context)
        else:  # ai mode
            return await self._execute_ai(input, context)

    async def _execute_deterministic(
        self,
        input: Any,
        context: ExecutionContext,
    ) -> NodeResult:
        """Execute deterministic condition evaluation.

        Args:
            input: Input data to evaluate
            context: Execution context

        Returns:
            NodeResult with routing information in metadata
        """
        # Emit start event
        await self._emit_event_if_enabled(
            context,
            ExecutionEvent(
                type=EventType.NODE_START,
                node_id=self.id,
                metadata={
                    "routing_mode": "deterministic",
                    "condition_count": len(self.conditions),
                    "node_type": "condition",
                },
            )
        )

        fulfilled_conditions = []
        unfulfilled_conditions = []

        # Evaluate each condition
        for condition in self.conditions:
            try:
                # Check if predicate accepts context (2 parameters)
                # by inspecting the function signature
                predicate = condition.predicate
                try:
                    sig = inspect.signature(predicate)
                    param_count = len(sig.parameters)
                except (ValueError, TypeError):
                    # Fallback for built-ins or other callables
                    param_count = 1

                # Call predicate with appropriate arguments
                if param_count >= 2:
                    # Predicate accepts input AND context
                    is_fulfilled = predicate(input, context)
                else:
                    # Predicate accepts only input
                    is_fulfilled = predicate(input)

                if is_fulfilled:
                    fulfilled_conditions.append(condition)
                else:
                    unfulfilled_conditions.append(condition)

            except Exception as e:
                # Treat evaluation errors as unfulfilled
                unfulfilled_conditions.append(condition)
                # Log the error
                print(f"Condition '{condition.name}' evaluation failed: {e}")

        # If no conditions fulfilled and default exists, use it
        if not fulfilled_conditions and self.default_target:
            fulfilled_conditions.append(
                Condition(
                    name="default",
                    predicate=lambda x: True,
                    target_node=self.default_target,
                )
            )

        # Build metadata for executor
        condition_metadata = []
        for condition in self.conditions:
            condition_metadata.append(
                {
                    "name": condition.name,
                    "target": condition.target_node,
                    "fulfilled": condition in fulfilled_conditions,
                }
            )

        # Add default if used
        if not any(c.name == "default" for c in self.conditions) and self.default_target:
            condition_metadata.append(
                {
                    "name": "default",
                    "target": self.default_target,
                    "fulfilled": not any(c in fulfilled_conditions for c in self.conditions),
                }
            )

        # Emit complete event
        branch_selected = fulfilled_conditions[0].target_node if fulfilled_conditions else self.default_target
        await self._emit_event_if_enabled(
            context,
            ExecutionEvent(
                type=EventType.NODE_COMPLETE,
                node_id=self.id,
                metadata={
                    "routing_mode": "deterministic",
                    "branch_selected": branch_selected,
                    "fulfilled_count": len(fulfilled_conditions),
                    "node_type": "condition",
                },
            )
        )

        return NodeResult(
            output={
                "input": input,  # Pass through input
                "fulfilled": [c.name for c in fulfilled_conditions],
                "unfulfilled": [c.name for c in unfulfilled_conditions],
            },
            metadata={
                "conditions": condition_metadata,
                "node_type": "condition",
                "routing_mode": "deterministic",
            },
        )

    async def _execute_ai(
        self,
        input: Any,
        context: ExecutionContext,
    ) -> NodeResult:
        """Execute AI-driven condition evaluation using LLM.

        Args:
            input: Input data to classify
            context: Execution context

        Returns:
            NodeResult with routing information based on LLM classification
        """
        # Emit start event
        await self._emit_event_if_enabled(
            context,
            ExecutionEvent(
                type=EventType.NODE_START,
                node_id=self.id,
                metadata={
                    "routing_mode": "ai",
                    "model": self.model,
                    "scenario_count": len(self.scenarios),
                    "node_type": "condition",
                },
            )
        )

        import json
        from openai import AsyncOpenAI

        # Initialize OpenAI client
        client = AsyncOpenAI()

        # Format input for LLM
        if isinstance(input, dict):
            input_text = json.dumps(input, indent=2)
        else:
            input_text = str(input)

        # Build scenario descriptions for prompt
        scenario_descriptions = "\n".join(
            [
                f"- {s['name']}: {s.get('description', s['name'])}"
                for s in self.scenarios
            ]
        )

        # Build system prompt
        system_prompt = f"""{self.instructions}

Available scenarios:
{scenario_descriptions}

Return ONLY the scenario name that best matches the input. Do not include any explanation."""

        # Call LLM
        try:
            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": input_text},
                ],
                temperature=0,  # Deterministic for consistent routing
                max_tokens=50,  # Short response expected
            )

            # Extract classification
            classification = response.choices[0].message.content.strip().lower()

            # Find matching scenario
            matched_scenario = None
            for scenario in self.scenarios:
                if scenario["name"].lower() == classification:
                    matched_scenario = scenario
                    break

            # Build metadata
            scenario_metadata = []
            for scenario in self.scenarios:
                scenario_metadata.append(
                    {
                        "name": scenario["name"],
                        "target": scenario["target"],
                        "fulfilled": scenario == matched_scenario,
                    }
                )

            # If no match and default exists, use default
            if not matched_scenario and self.default_target:
                scenario_metadata.append(
                    {
                        "name": "default",
                        "target": self.default_target,
                        "fulfilled": True,
                    }
                )
                fulfilled_scenario = "default"
                target = self.default_target
            elif matched_scenario:
                fulfilled_scenario = matched_scenario["name"]
                target = matched_scenario["target"]
            else:
                # No match and no default - error
                raise ValueError(
                    f"LLM returned '{classification}' which doesn't match any scenario"
                )

            # Emit complete event
            await self._emit_event_if_enabled(
                context,
                ExecutionEvent(
                    type=EventType.NODE_COMPLETE,
                    node_id=self.id,
                    metadata={
                        "routing_mode": "ai",
                        "classification": classification,
                        "branch_selected": target,
                        "node_type": "condition",
                    },
                )
            )

            return NodeResult(
                output={
                    "input": input,
                    "classification": classification,
                    "scenario": fulfilled_scenario,
                    "target": target,
                },
                metadata={
                    "scenarios": scenario_metadata,
                    "node_type": "condition",
                    "routing_mode": "ai",
                    "llm_response": classification,
                },
            )

        except Exception as e:
            # LLM call failed - use default if available
            if self.default_target:
                return NodeResult(
                    output={
                        "input": input,
                        "classification": "error",
                        "scenario": "default",
                        "target": self.default_target,
                        "error": str(e),
                    },
                    metadata={
                        "scenarios": [
                            {
                                "name": "default",
                                "target": self.default_target,
                                "fulfilled": True,
                            }
                        ],
                        "node_type": "condition",
                        "routing_mode": "ai",
                        "error": str(e),
                    },
                )
            else:
                raise RuntimeError(f"AI condition routing failed: {e}") from e

    def __repr__(self) -> str:
        if self.condition_routing == "deterministic":
            return f"ConditionNode(id='{self.id}', mode='deterministic', conditions={len(self.conditions)})"
        else:
            return f"ConditionNode(id='{self.id}', mode='ai', model='{self.model}', scenarios={len(self.scenarios)})"


class SimpleCondition:
    """Helper class for creating conditions with simple comparison logic.

    This class provides static methods that generate Condition objects for
    common comparison operations, matching Flowise's Condition Node operations.

    All methods follow the pattern: (field, value, target_node) → Condition

    Available Operations:
        - equals(field, value, target): field == value
        - not_equal(field, value, target): field != value
        - contains(field, substring, target): substring in field
        - not_contains(field, substring, target): substring not in field
        - greater_than(field, threshold, target): field > threshold
        - less_than(field, threshold, target): field < threshold
        - is_empty(field, target): field is None or len(field) == 0

    Example:
        >>> from mesh.nodes import ConditionNode, SimpleCondition
        >>>
        >>> # Create condition node with multiple operations
        >>> router = ConditionNode(
        ...     id="router",
        ...     conditions=[
        ...         SimpleCondition.equals("status", "approved", "approve_path"),
        ...         SimpleCondition.contains("message", "urgent", "urgent_path"),
        ...         SimpleCondition.greater_than("priority", 5, "high_priority_path"),
        ...         SimpleCondition.is_empty("errors", "success_path"),
        ...     ]
        ... )

    Note:
        All conditions expect input to be a dictionary. If input is not a dict,
        the predicate will return False.
    """

    @staticmethod
    def equals(field: str, value: Any, target_node: str) -> Condition:
        """Create condition that checks if field equals value."""

        def predicate(input: Any) -> bool:
            if isinstance(input, dict):
                return input.get(field) == value
            return False

        return Condition(
            name=f"{field}_equals_{value}",
            predicate=predicate,
            target_node=target_node,
        )

    @staticmethod
    def contains(field: str, substring: str, target_node: str) -> Condition:
        """Create condition that checks if field contains substring."""

        def predicate(input: Any) -> bool:
            if isinstance(input, dict):
                value = str(input.get(field, ""))
                return substring in value
            return False

        return Condition(
            name=f"{field}_contains_{substring}",
            predicate=predicate,
            target_node=target_node,
        )

    @staticmethod
    def greater_than(field: str, threshold: float, target_node: str) -> Condition:
        """Create condition that checks if field is greater than threshold."""

        def predicate(input: Any) -> bool:
            if isinstance(input, dict):
                try:
                    value = float(input.get(field, 0))
                    return value > threshold
                except (ValueError, TypeError):
                    return False
            return False

        return Condition(
            name=f"{field}_gt_{threshold}",
            predicate=predicate,
            target_node=target_node,
        )

    @staticmethod
    def not_equal(field: str, value: Any, target_node: str) -> Condition:
        """Create condition that checks if field does not equal value."""

        def predicate(input: Any) -> bool:
            if isinstance(input, dict):
                return input.get(field) != value
            return False

        return Condition(
            name=f"{field}_not_equals_{value}",
            predicate=predicate,
            target_node=target_node,
        )

    @staticmethod
    def less_than(field: str, threshold: float, target_node: str) -> Condition:
        """Create condition that checks if field is less than threshold."""

        def predicate(input: Any) -> bool:
            if isinstance(input, dict):
                try:
                    value = float(input.get(field, 0))
                    return value < threshold
                except (ValueError, TypeError):
                    return False
            return False

        return Condition(
            name=f"{field}_lt_{threshold}",
            predicate=predicate,
            target_node=target_node,
        )

    @staticmethod
    def is_empty(field: str, target_node: str) -> Condition:
        """Create condition that checks if field is empty (None, empty string, empty list, etc.)."""

        def predicate(input: Any) -> bool:
            if isinstance(input, dict):
                value = input.get(field)
                # Check for various empty states
                if value is None:
                    return True
                if isinstance(value, (str, list, dict)):
                    return len(value) == 0
                return False
            return False

        return Condition(
            name=f"{field}_is_empty",
            predicate=predicate,
            target_node=target_node,
        )

    @staticmethod
    def not_contains(field: str, substring: str, target_node: str) -> Condition:
        """Create condition that checks if field does not contain substring."""

        def predicate(input: Any) -> bool:
            if isinstance(input, dict):
                value = str(input.get(field, ""))
                return substring not in value
            return False

        return Condition(
            name=f"{field}_not_contains_{substring}",
            predicate=predicate,
            target_node=target_node,
        )

from anthropic import Anthropic
from .tools import *
from dotenv import load_dotenv
from core.logging.logger import app_logger, StructuredAppLog
import json
import time
import random
import re
import logging
from typing import Any, Dict, List, Optional, Union
from enum import Enum

load_dotenv()

# Configure module logger
logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """
    Circuit breaker pattern implementation to prevent cascade failures.

    Prevents continuous retries when the API is consistently failing,
    allowing the system to fail fast and recover gracefully.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 1
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
            half_open_max_calls: Max calls allowed in half-open state
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.half_open_calls = 0

    def can_execute(self) -> bool:
        """Check if request can be executed based on circuit state."""
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if self.last_failure_time and \
               time.time() - self.last_failure_time >= self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                self.half_open_calls = 0
                logger.info("Circuit breaker transitioning to HALF_OPEN state")
                return True
            return False

        if self.state == CircuitState.HALF_OPEN:
            return self.half_open_calls < self.half_open_max_calls

        return False

    def record_success(self) -> None:
        """Record a successful call."""
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            logger.info("Circuit breaker recovered - transitioning to CLOSED state")

        self.failure_count = 0
        self.half_open_calls = 0

    def record_failure(self) -> None:
        """Record a failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            logger.warning("Circuit breaker re-opened after failure in HALF_OPEN state")

        elif self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(
                f"Circuit breaker OPEN after {self.failure_count} failures. "
                f"Will retry after {self.recovery_timeout}s"
            )

    def get_state_info(self) -> Dict[str, Any]:
        """Get current circuit breaker state information."""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "last_failure_time": self.last_failure_time,
            "time_until_retry": max(
                0,
                self.recovery_timeout - (time.time() - (self.last_failure_time or 0))
            ) if self.state == CircuitState.OPEN else 0
        }


class RetryableError(Exception):
    """Exception indicating an error that can be retried."""
    pass


class NonRetryableError(Exception):
    """Exception indicating an error that should not be retried."""
    pass

IDENTITY = """You are Halberd Attack Agent, a helpful and knowledgeable AI assistant for Halberd Attack Tool. 
Your role is to assist Halberd tool users, provide information related to tool capabilities, help plan cloud security testing and execute Halberd attack techniques. 
Generate all your responses properly formatted for markdown rendering. 

STEPS TO EXECUTE A TECHNIQUE: 
1. Get appropriate technique
2. Check technique's input requirements
3. Based on technique inputs, configure technique with appropriate inputs. Ask user for input if required.
4. CONFIRM WITH THE USER THE FINAL TECHNIQUE CONFIGURATION
5. Execute technique
6. Return technique execution response in the following structure: 
<technique-response-format>
# Technique Name
Result : {Success(green checkmark)/ Failed(Red X)}

Event ID : {event-id}

## Output Analysis
- Brief analysis of the technique output in bullet points. 
If the output is too large, display a short note to check the full output in [attack-history](/attack-history)

Summary information of the output (if applicable)

## Recommendations
- Table of technique names for next step recommendation if applicable
End response
</technique-response-format>
"""

tool_functions = {
    "execute_technique":execute_technique,
    "list_techniques":list_techniques,
    "list_tactics":list_tactics,
    "get_technique_mitre_info": get_technique_mitre_info,
    "get_technique_aztrm_info": get_technique_aztrm_info,
    "get_technique_inputs": get_technique_inputs,
    "entra_id_get_all_tokens": entra_id_get_all_tokens,
    "entra_id_get_active_token": entra_id_get_active_token,
    "entra_id_get_active_token_pair": entra_id_get_active_token_pair,
    "entra_id_set_active_token": entra_id_set_active_token,
    "entra_id_decode_jwt_token": entra_id_decode_jwt_token,
    "aws_get_all_sessions": aws_get_all_sessions,
    "aws_retrieve_sessions": aws_retrieve_sessions,
    "aws_get_active_session": aws_get_active_session,
    "aws_get_session_details": aws_get_session_details,
    "aws_set_active_session": aws_set_active_session,
    "aws_get_connected_user_details": aws_get_connected_user_details,
    "read_halberd_logs": read_halberd_logs,
    "get_technique_execution_response": get_technique_execution_response,
    "get_app_info": get_app_info
}

# Token limits
MAX_TOTAL_TOKENS = 200000  # Anthropic's total token limit
MAX_MODEL_TOKENS = 4096  # Claude-3-7-sonnet max tokens
MAX_TOOL_RESPONSE_TOKENS = 15000  # Max size for tool responses
TOKEN_WARNING_THRESHOLD = 0.8  # Warn user when reaching 80% of token limit

# Rate limiting parameters
RATE_LIMIT_TOKENS_PER_MIN = 20000  # Anthropic's rate limit (tokens per minute)
OUTPUT_RATE_LIMIT_TOKENS_PER_MIN = 8000  # Output tokens rate limit
MIN_DELAY_BETWEEN_CALLS = 3  # Minimum delay in seconds between API calls

# Retry configuration
MAX_RETRIES = 5
BASE_RETRY_DELAY = 2  # Base delay in seconds for exponential backoff
API_TIMEOUT = 120  # Timeout in seconds for API calls

# API key validation pattern (Anthropic API keys start with 'sk-ant-')
API_KEY_PATTERN = re.compile(r'^sk-ant-[a-zA-Z0-9-_]{40,}$')

# Input validation limits
MAX_USER_MESSAGE_LENGTH = 100000  # Max characters for user input
MAX_ATTACHMENT_SIZE = 10 * 1024 * 1024  # 10MB max attachment size

# Retryable HTTP error codes
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

# Retryable error patterns
RETRYABLE_ERROR_PATTERNS = [
    "rate_limit",
    "overloaded",
    "timeout",
    "connection",
    "network",
    "temporarily unavailable",
    "service unavailable",
    "internal server error",
    "bad gateway",
    "gateway timeout"
]

class RateLimiter:
    """Tracks API usage and calculates necessary delays to avoid rate limits."""
    
    def __init__(self):
        self.requests_timestamps = []  # Track request timestamps
        self.input_token_usage = []    # Track (timestamp, input_token_count)
        self.output_token_usage = []   # Track (timestamp, output_token_count)
        
    def _clean_old_records(self, current_time):
        """Remove records older than 60 seconds from all tracking lists."""
        minute_ago = current_time - 60
        
        # Clean request timestamps
        self.requests_timestamps = [t for t in self.requests_timestamps if t > minute_ago]
        
        # Clean token usage records
        self.input_token_usage = [(t, count) for t, count in self.input_token_usage if t > minute_ago]
        self.output_token_usage = [(t, count) for t, count in self.output_token_usage if t > minute_ago]
    
    def _calculate_request_delay(self, current_time):
        """Calculate delay needed to stay under request rate limit."""
        # If we haven't hit the request limit, no delay needed
        if len(self.requests_timestamps) < 50:
            return 0
            
        # Calculate when the oldest request will expire from the window
        oldest_timestamp = min(self.requests_timestamps)
        time_until_slot_available = (oldest_timestamp + 60) - current_time
        
        return max(0, time_until_slot_available)
    
    def _calculate_token_delay(self, new_tokens, usage_records, limit, current_time):
        """Calculate delay needed to stay under token rate limit."""
        # Sum current usage in the rolling window
        current_usage = sum(count for _, count in usage_records)
        
        # If adding new tokens doesn't exceed limit, no delay needed
        if current_usage + new_tokens <= limit:
            return 0
            
        # Calculate tokens that need to expire before we can proceed
        tokens_to_expire = (current_usage + new_tokens) - limit
        
        # Sort usage records by timestamp (oldest first)
        sorted_records = sorted(usage_records, key=lambda x: x[0])
        
        # Find wait time for enough tokens to expire
        tokens_expired = 0
        for timestamp, count in sorted_records:
            tokens_expired += count
            if tokens_expired >= tokens_to_expire:
                return max(0, (timestamp + 60) - current_time)
                
        # If we can't expire enough tokens within the window, use maximum delay
        return 60  # Maximum possible delay
    
    def get_required_delay(self, input_tokens, est_output_tokens):
        """Calculate the delay required to satisfy all rate limits."""
        current_time = time.time()
        
        # Remove records older than 60 seconds
        self._clean_old_records(current_time)
        
        # Calculate delays for each constraint
        request_delay = self._calculate_request_delay(current_time)
        input_token_delay = self._calculate_token_delay(
            input_tokens, self.input_token_usage, RATE_LIMIT_TOKENS_PER_MIN, current_time)
        output_token_delay = self._calculate_token_delay(
            est_output_tokens, self.output_token_usage, OUTPUT_RATE_LIMIT_TOKENS_PER_MIN, current_time)
        
        # Get maximum delay required
        max_delay = max(request_delay, input_token_delay, output_token_delay)
        
        if max_delay > 0:
            reason = "requests"
            if input_token_delay == max_delay:
                reason = "input tokens"
            elif output_token_delay == max_delay:
                reason = "output tokens"
            logger.info(f"Rate limiting: Delay of {max_delay:.2f}s required due to {reason} limit")

        return max_delay
    
    def record_usage(self, input_tokens, output_tokens, timestamp=None):
        """Record actual usage after an API call."""
        if timestamp is None:
            timestamp = time.time()
            
        self.requests_timestamps.append(timestamp)
        self.input_token_usage.append((timestamp, input_tokens))
        self.output_token_usage.append((timestamp, output_tokens))

class AttackAgent:
    """
    AI-powered attack agent for Halberd security testing tool.

    Manages conversations with Anthropic's Claude API, handles tool execution,
    and provides robust error handling with retry logic and circuit breaker patterns.
    """

    def __init__(self, session_state):
        """
        Initialize the AttackAgent.

        Args:
            session_state: Session state object for managing conversation history
        """
        self.session_state = session_state
        self.last_api_call_time = 0  # Track last API call time for rate limiting

        # Token tracking variables
        self.current_conversation_input_tokens = 0
        self.current_conversation_output_tokens = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

        # Client & api key variables
        self._anthropic_client = None
        self._last_api_key = None

        # Initialize rate limiter and circuit breaker
        self.rate_limiter = RateLimiter()
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60.0,
            half_open_max_calls=1
        )

        # Track token usage warnings
        self._token_warning_issued = False

        logger.info("AttackAgent initialized")

    @property
    def anthropic(self) -> Optional[Anthropic]:
        """
        Lazy initialization of Anthropic client that updates when API key changes.

        Returns:
            Anthropic client instance or None if not available
        """
        load_dotenv()
        current_api_key = os.environ.get('ANTHROPIC_API_KEY')

        # If no client yet or the API key has changed -> create a new one
        if (self._anthropic_client is None or
            current_api_key != self._last_api_key):

            if current_api_key:
                # Validate API key format
                validation_result = self._validate_api_key(current_api_key)
                if not validation_result["valid"]:
                    logger.warning(f"API key validation warning: {validation_result['message']}")

                try:
                    self._anthropic_client = Anthropic(timeout=API_TIMEOUT)
                    self._last_api_key = current_api_key
                    logger.info("Anthropic client initialized successfully")
                except Exception as e:
                    logger.error(f"Failed to initialize Anthropic client: {e}")
                    self._anthropic_client = None
            else:
                logger.warning("ANTHROPIC_API_KEY environment variable not set")
                self._anthropic_client = None

        return self._anthropic_client

    def _validate_api_key(self, api_key: str) -> Dict[str, Any]:
        """
        Validate Anthropic API key format.

        Args:
            api_key: The API key to validate

        Returns:
            Dictionary with 'valid' boolean and 'message' string
        """
        if not api_key:
            return {"valid": False, "message": "API key is empty"}

        if not api_key.startswith('sk-ant-'):
            return {
                "valid": False,
                "message": "API key should start with 'sk-ant-'. Please verify your API key."
            }

        if len(api_key) < 40:
            return {
                "valid": False,
                "message": "API key appears too short. Please verify your API key."
            }

        # Basic pattern check (warn but don't block)
        if not API_KEY_PATTERN.match(api_key):
            return {
                "valid": True,
                "message": "API key format is unusual but may still work."
            }

        return {"valid": True, "message": "API key format is valid"}

    def is_anthropic_ready(self) -> bool:
        """
        Check if Anthropic client is ready to use.

        Returns:
            True if client is initialized and ready
        """
        return self.anthropic is not None

    def _is_retryable_error(self, error: Exception) -> bool:
        """
        Determine if an error should be retried.

        Args:
            error: The exception that occurred

        Returns:
            True if the error is retryable
        """
        error_str = str(error).lower()

        # Check for retryable patterns
        for pattern in RETRYABLE_ERROR_PATTERNS:
            if pattern in error_str:
                return True

        # Check for HTTP status codes in error message
        for code in RETRYABLE_STATUS_CODES:
            if str(code) in str(error):
                return True

        return False

    def _validate_user_input(self, user_input: Any) -> Dict[str, Any]:
        """
        Validate user input before processing.

        Args:
            user_input: The user's input message

        Returns:
            Dictionary with 'valid' boolean and 'message' string
        """
        if user_input is None:
            return {"valid": False, "message": "Input cannot be None"}

        # Handle string input
        if isinstance(user_input, str):
            if len(user_input) == 0:
                return {"valid": False, "message": "Input cannot be empty"}

            if len(user_input) > MAX_USER_MESSAGE_LENGTH:
                return {
                    "valid": False,
                    "message": f"Input exceeds maximum length of {MAX_USER_MESSAGE_LENGTH} characters"
                }

        # Handle list input (multimodal)
        elif isinstance(user_input, list):
            if len(user_input) == 0:
                return {"valid": False, "message": "Input list cannot be empty"}

            for item in user_input:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        text = item.get("text", "")
                        if len(text) > MAX_USER_MESSAGE_LENGTH:
                            return {
                                "valid": False,
                                "message": f"Text content exceeds maximum length"
                            }
                    elif item.get("type") == "image":
                        # Validate image data if present
                        source = item.get("source", {})
                        if source.get("type") == "base64":
                            data = source.get("data", "")
                            if len(data) > MAX_ATTACHMENT_SIZE:
                                return {
                                    "valid": False,
                                    "message": "Image attachment exceeds maximum size"
                                }

        return {"valid": True, "message": "Input is valid"}

    def _check_token_warning(self) -> Optional[str]:
        """
        Check if token usage is approaching limits and return warning if needed.

        Returns:
            Warning message if threshold exceeded, None otherwise
        """
        total_tokens = self.current_conversation_input_tokens + self.current_conversation_output_tokens
        threshold = int(MAX_TOTAL_TOKENS * TOKEN_WARNING_THRESHOLD)

        if total_tokens >= threshold and not self._token_warning_issued:
            self._token_warning_issued = True
            remaining = MAX_TOTAL_TOKENS - total_tokens
            return (
                f"⚠️ **Token Usage Warning**: You've used {total_tokens:,} tokens "
                f"({remaining:,} remaining). Consider starting a new conversation "
                f"to avoid message truncation."
            )

        return None

    def count_tokens_for_messages(
        self,
        messages: List[Dict[str, Any]],
        include_system: bool = True
    ) -> int:
        """
        Count tokens for a complete message array using Anthropic's native counting.

        Args:
            messages: List of message dictionaries
            include_system: Whether to include system message in count

        Returns:
            Token count for the messages
        """
        if not self.is_anthropic_ready():
            # Fallback estimation if client not ready
            return self._estimate_tokens_fallback(messages, include_system)

        try:
            # Prepare messages for token counting
            messages_for_counting = []

            # Add conversation messages
            for msg in messages:
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    messages_for_counting.append(msg)
                else:
                    # Convert non-standard message format
                    messages_for_counting.append({
                        "role": "user",
                        "content": json.dumps(msg) if not isinstance(msg, str) else str(msg)
                    })

            # Count tokens using Anthropic's native method
            if include_system:
                # Include system message in counting
                count_result = self.anthropic.messages.count_tokens(
                    model="claude-3-7-sonnet-20250219",
                    messages=messages_for_counting,
                    system=IDENTITY
                )
            else:
                # Count only conversation messages
                count_result = self.anthropic.messages.count_tokens(
                    model="claude-3-7-sonnet-20250219",
                    messages=messages_for_counting
                )

            return count_result.input_tokens

        except Exception as e:
            logger.warning(f"Token counting error, using fallback: {e}")
            return self._estimate_tokens_fallback(messages, include_system)

    def count_tokens_for_content(self, content: Union[str, Any]) -> int:
        """
        Count tokens for arbitrary content by converting to message format.

        Args:
            content: Content to count tokens for (string or other serializable)

        Returns:
            Token count for the content
        """
        if not self.is_anthropic_ready():
            # Fallback estimation
            return self._estimate_content_tokens(content)

        try:
            # Convert content to message format
            if isinstance(content, str):
                messages = [{"role": "user", "content": content}]
            else:
                messages = [{"role": "user", "content": str(content)}]

            # Count tokens without system message (since this is just content)
            count_result = self.anthropic.messages.count_tokens(
                model="claude-3-7-sonnet-20250219",
                messages=messages
            )

            return count_result.input_tokens

        except Exception as e:
            logger.warning(f"Content token counting error, using fallback: {e}")
            return self._estimate_content_tokens(content)

    def _estimate_content_tokens(self, content: Any) -> int:
        """
        Estimate tokens for content when native counting unavailable.

        Uses a more accurate estimation based on typical tokenization patterns.

        Args:
            content: Content to estimate tokens for

        Returns:
            Estimated token count
        """
        if isinstance(content, str):
            text = content
        else:
            text = str(content)

        # More accurate estimation:
        # - Average English word is ~4-5 characters
        # - Average token is ~4 characters for common text
        # - Code tends to have more tokens per character
        # - JSON structure adds overhead

        # Count words and special characters
        words = len(text.split())
        special_chars = sum(1 for c in text if c in '{}[](),:;"\'<>=/\\@#$%^&*')

        # Estimate: words + special characters (which often become separate tokens)
        # Plus a base overhead for structure
        estimated = words + (special_chars // 2) + 10

        # Also consider raw character count as a minimum
        char_estimate = len(text) // 4

        # Use the higher estimate for safety
        return max(estimated, char_estimate)

    def _estimate_tokens_fallback(
        self,
        content: Union[List, Dict, str, Any],
        include_system: bool = False
    ) -> int:
        """
        Fallback token estimation when native counting is unavailable.

        Args:
            content: Content to estimate tokens for
            include_system: Whether to include system message in estimate

        Returns:
            Estimated token count
        """
        total_chars = 0

        if include_system:
            total_chars += len(IDENTITY)

        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    total_chars += len(json.dumps(item))
                else:
                    total_chars += len(str(item))
        elif isinstance(content, dict):
            total_chars += len(json.dumps(content))
        else:
            total_chars += len(str(content))

        # Use improved estimation
        return self._estimate_content_tokens(" " * total_chars)

    def truncate_messages_history(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int
    ) -> List[Dict[str, Any]]:
        """
        Truncate message history to fit within token limits.

        Args:
            messages: List of messages to truncate
            max_tokens: Maximum token limit

        Returns:
            Truncated message list
        """
        # Always keep at least the latest user message
        if len(messages) <= 1:
            return messages

        original_count = len(messages)

        # Start with the most recent message
        recent_messages = [messages[-1]]

        # Calculate initial token count (including system message)
        token_count = self.count_tokens_for_messages(recent_messages, include_system=True)

        # Add messages from newest to oldest until we approach the limit
        for message in reversed(messages[:-1]):
            # Calculate tokens for this message
            temp_messages = [message] + recent_messages
            temp_token_count = self.count_tokens_for_messages(temp_messages, include_system=True)

            if temp_token_count < max_tokens:
                recent_messages.insert(0, message)
                token_count = temp_token_count
            else:
                break

        truncated_count = original_count - len(recent_messages)
        if truncated_count > 0:
            logger.info(
                f"Truncated {truncated_count} messages from history "
                f"(kept {len(recent_messages)}/{original_count})"
            )

        return recent_messages

    def generate_message(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int
    ) -> Union[Any, Dict[str, str]]:
        """
        Generate a message using Anthropic API with robust error handling.

        Implements:
        - Rate limiting with rolling windows
        - Circuit breaker pattern
        - Exponential backoff retry for transient errors
        - Comprehensive error classification

        Args:
            messages: List of conversation messages
            max_tokens: Maximum tokens for response

        Returns:
            API response object or error dictionary
        """
        # Check circuit breaker state
        if not self.circuit_breaker.can_execute():
            state_info = self.circuit_breaker.get_state_info()
            wait_time = int(state_info["time_until_retry"])
            logger.warning(f"Circuit breaker is OPEN. Retry in {wait_time}s")
            return {
                "error": f"Service temporarily unavailable. Please retry in {wait_time} seconds. "
                         f"(Circuit breaker triggered after repeated failures)"
            }

        # Count total tokens including system message
        total_tokens = self.count_tokens_for_messages(messages, include_system=True)

        # Track input tokens for this call
        input_tokens = total_tokens
        self.current_conversation_input_tokens += input_tokens
        self.total_input_tokens += input_tokens

        if total_tokens > MAX_TOTAL_TOKENS - max_tokens:
            # Truncate history to fit within limits
            messages = self.truncate_messages_history(messages, MAX_TOTAL_TOKENS - max_tokens)
            # Recalculate tokens after truncation
            new_total_tokens = self.count_tokens_for_messages(messages, include_system=True)

            # Adjust token counts
            token_reduction = total_tokens - new_total_tokens
            self.current_conversation_input_tokens -= token_reduction
            self.total_input_tokens -= token_reduction

            total_tokens = new_total_tokens

        # Estimate output tokens (about 75% of their max_tokens allocation)
        estimated_output_tokens = int(max_tokens * 0.75)

        # Get required delay from rate limiter
        required_delay = self.rate_limiter.get_required_delay(total_tokens, estimated_output_tokens)

        # Apply delay if needed (either from rate limits or minimum delay)
        if required_delay > 0:
            time.sleep(required_delay)
        else:
            # Ensure minimum delay between consecutive calls
            time_since_last_call = time.time() - self.last_api_call_time
            if time_since_last_call < MIN_DELAY_BETWEEN_CALLS and self.last_api_call_time > 0:
                minimum_delay = MIN_DELAY_BETWEEN_CALLS - time_since_last_call
                time.sleep(minimum_delay)
                logger.debug(f"Enforcing minimum delay: {minimum_delay:.2f}s between API calls")

        # Update last API call time before making the call
        self.last_api_call_time = time.time()

        last_error = None

        for attempt in range(MAX_RETRIES):
            try:
                call_start_time = time.time()

                logger.debug(f"API call attempt {attempt + 1}/{MAX_RETRIES}")

                # Make the API call
                response = self.anthropic.messages.create(
                    model="claude-3-7-sonnet-20250219",
                    system=[
                        {
                            "type": "text",
                            "text": IDENTITY,
                            "cache_control": {"type": "ephemeral"}
                        }
                    ],
                    max_tokens=max_tokens,
                    messages=messages,
                    tools=tools,
                )

                # Count tokens in the actual response
                actual_output_tokens = self.count_tokens_in_response(response)

                # Track output tokens for this call
                self.current_conversation_output_tokens += actual_output_tokens
                self.total_output_tokens += actual_output_tokens

                # Record usage for rate limiting
                self.rate_limiter.record_usage(total_tokens, actual_output_tokens, call_start_time)

                # Record success with circuit breaker
                self.circuit_breaker.record_success()

                logger.debug(
                    f"API call successful. Tokens: input={total_tokens}, output={actual_output_tokens}"
                )

                return response

            except Exception as e:
                last_error = e
                error_str = str(e)

                # Determine if error is retryable
                is_retryable = self._is_retryable_error(e)

                if is_retryable:
                    # If this is the last retry, fail
                    if attempt == MAX_RETRIES - 1:
                        self.circuit_breaker.record_failure()
                        logger.error(f"All {MAX_RETRIES} retry attempts failed: {error_str}")
                        return {
                            "error": f"Request failed after {MAX_RETRIES} attempts: {error_str}. "
                                     "Please try again later."
                        }

                    # Calculate backoff delay with jitter
                    delay = BASE_RETRY_DELAY * (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(
                        f"Retryable error occurred: {error_str}. "
                        f"Retrying in {delay:.2f}s (attempt {attempt + 1}/{MAX_RETRIES})"
                    )
                    time.sleep(delay)

                    # After a rate limit, update our internal tracking to be more cautious
                    if "rate_limit" in error_str.lower() or "429" in error_str:
                        now = time.time()
                        cautious_input_tokens = int(total_tokens * 0.5)
                        cautious_output_tokens = int(estimated_output_tokens * 0.5)
                        self.rate_limiter.record_usage(
                            cautious_input_tokens,
                            cautious_output_tokens,
                            now - 1
                        )
                else:
                    # Non-retryable error - fail immediately
                    self.circuit_breaker.record_failure()
                    logger.error(f"Non-retryable error: {error_str}")

                    # Provide user-friendly error messages
                    if "authentication" in error_str.lower() or "api_key" in error_str.lower():
                        return {
                            "error": "Authentication failed. Please verify your ANTHROPIC_API_KEY "
                                     "is correctly set in your environment."
                        }
                    elif "invalid_request" in error_str.lower():
                        return {
                            "error": f"Invalid request: {error_str}. Please check your input."
                        }
                    else:
                        return {"error": error_str}

        # Should not reach here, but handle gracefully
        self.circuit_breaker.record_failure()
        return {"error": f"Unexpected error after retries: {str(last_error)}"}
    
    def process_user_input(self, user_input: Any) -> str:
        """
        Process user input, manage tool calls, and maintain a valid conversation structure.

        Args:
            user_input: User's message (string or multimodal content)

        Returns:
            Response text from the AI agent
        """
        # Validate user input
        validation_result = self._validate_user_input(user_input)
        if not validation_result["valid"]:
            error_msg = f"Input validation error: {validation_result['message']}"
            logger.warning(error_msg)
            return error_msg

        # Check for token usage warning
        token_warning = self._check_token_warning()

        # Create main user message
        self.session_state.messages.append({"role": "user", "content": user_input})

        logger.info(f"Processing user input (length: {len(str(user_input))} chars)")

        try:
            # Initial model response
            response_message = self.generate_message(
                messages=self.session_state.messages,
                max_tokens=MAX_MODEL_TOKENS,
            )

            if isinstance(response_message, dict) and "error" in response_message:
                error_text = f"An error occurred: {response_message['error']}"
                logger.error(error_text)
                return error_text
            
            # Keep processing tool calls until there are none left
            current_message = response_message

            while any(content.type == "tool_use" for content in current_message.content):
                # Get all tool calls in this message
                tool_calls = [content for content in current_message.content if content.type == "tool_use"]
                logger.info(f"Processing {len(tool_calls)} tool call(s): {[tc.name for tc in tool_calls]}")

                # Convert message content to serializable format before adding to history
                serializable_content = []
                for content_block in current_message.content:
                    if content_block.type == "text":
                        serializable_content.append({"type": "text", "text": content_block.text})
                    elif content_block.type == "tool_use":
                        serializable_content.append({
                            "type": "tool_use",
                            "id": content_block.id,
                            "name": content_block.name,
                            "input": content_block.input
                        })

                # Add the serialized message to the conversation history
                self.session_state.messages.append(
                    {"role": "assistant", "content": serializable_content}
                )

                # Process each tool call
                tool_results = []
                for tool_use in tool_calls:
                    func_name = tool_use.name
                    func_params = tool_use.input
                    tool_use_id = tool_use.id

                    # Execute tool and get result
                    result = self.handle_tool_use(func_name, func_params)

                    # Check if tool response is too large using content token counting
                    result_str = str(result)
                    result_tokens = self.count_tokens_for_content(result_str)

                    if result_tokens > MAX_TOOL_RESPONSE_TOKENS:
                        # Smart truncation with context preservation
                        result_content = self._smart_truncate_tool_response(
                            result_str, result_tokens, MAX_TOOL_RESPONSE_TOKENS
                        )
                        logger.warning(
                            f"Tool '{func_name}' response truncated: "
                            f"{result_tokens} tokens -> {MAX_TOOL_RESPONSE_TOKENS} tokens"
                        )
                    else:
                        result_content = result_str

                    # Add tool_result for each tool_use
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": result_content,
                    })

                # Add tool results to the conversation
                self.session_state.messages.append({
                    "role": "user",
                    "content": tool_results,
                })

                try:
                    # Get the next response from the model
                    current_message = self.generate_message(
                        messages=self.session_state.messages,
                        max_tokens=MAX_MODEL_TOKENS,
                    )

                    if isinstance(current_message, dict) and "error" in current_message:
                        error_text = f"An error occurred: {current_message['error']}"
                        logger.error(error_text)
                        return error_text

                except Exception as e:
                    # If there's an error getting the next message, add error message
                    error_message = f"Error generating response: {str(e)}"
                    logger.error(error_message)
                    self.session_state.messages.append({
                        "role": "assistant",
                        "content": [{"type": "text", "text": error_message}]
                    })
                    return error_message
            
            # Extract response text from final message
            response_text = ''.join([content.text for content in current_message.content if content.type == "text"])

            # Convert final message content to serializable format
            serializable_content = []
            for content_block in current_message.content:
                if content_block.type == "text":
                    serializable_content.append({"type": "text", "text": content_block.text})
                elif content_block.type == "tool_use":
                    serializable_content.append({
                        "type": "tool_use",
                        "id": content_block.id,
                        "name": content_block.name,
                        "input": content_block.input
                    })

            # Add the final message to conversation history
            self.session_state.messages.append(
                {"role": "assistant", "content": serializable_content}
            )

            logger.info("Successfully processed user input and generated response")

            # Prepend token warning if applicable
            if token_warning:
                response_text = f"{token_warning}\n\n---\n\n{response_text}"

            return response_text

        except Exception as e:
            # Catch any unexpected errors during processing
            error_message = f"An unexpected error occurred: {str(e)}"
            logger.error(f"Unexpected error in process_user_input: {e}", exc_info=True)
            self.session_state.messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": error_message}]
            })
            return error_message

    def _smart_truncate_tool_response(
        self,
        response: str,
        current_tokens: int,
        max_tokens: int
    ) -> str:
        """
        Intelligently truncate tool response while preserving useful information.

        Args:
            response: The full tool response
            current_tokens: Current token count of response
            max_tokens: Maximum allowed tokens

        Returns:
            Truncated response with context
        """
        # Calculate target character count (rough estimate)
        ratio = max_tokens / current_tokens
        target_chars = int(len(response) * ratio * 0.9)  # 90% to be safe

        # Reserve space for metadata
        metadata_reserve = 500
        available_chars = target_chars - metadata_reserve

        if available_chars <= 0:
            return (
                f"Tool response exceeded token limit ({current_tokens:,} tokens, "
                f"max {max_tokens:,}). Response too large to display. "
                "Please check the full output in attack history."
            )

        # Try to find a good truncation point
        truncated = response[:available_chars]

        # Try to end at a natural break point
        break_points = ['\n\n', '\n', '. ', ', ', ' ']
        for bp in break_points:
            last_break = truncated.rfind(bp)
            if last_break > available_chars * 0.8:  # Keep at least 80%
                truncated = truncated[:last_break + len(bp)]
                break

        # Build the truncated response with metadata
        lines_total = response.count('\n') + 1
        lines_shown = truncated.count('\n') + 1

        result = (
            f"⚠️ **Response Truncated** (showing ~{len(truncated):,} of {len(response):,} characters)\n\n"
            f"{truncated}\n\n"
            f"---\n"
            f"*[Truncated: {current_tokens:,} tokens exceeded {max_tokens:,} limit. "
            f"Showing {lines_shown}/{lines_total} lines. "
            f"Full output available in attack history.]*"
        )

        return result
   
    def handle_tool_use(self, tool_name: str, tool_input: Dict[str, Any]) -> Any:
        """
        Execute a tool call and handle any exceptions.

        Args:
            tool_name: Name of the tool to execute
            tool_input: Dictionary of input parameters for the tool

        Returns:
            Tool execution result or error message
        """
        logger.info(f"Executing tool: {tool_name}")
        logger.debug(f"Tool input: {tool_input}")

        try:
            if tool_name in tool_functions:
                result = tool_functions[tool_name](**tool_input)
                logger.info(f"Tool '{tool_name}' executed successfully")
                return result
            else:
                error_msg = f"Error: Tool '{tool_name}' not found in available tools"
                logger.error(error_msg)
                return error_msg

        except TypeError as e:
            # Handle missing or invalid parameters
            error_msg = f"Error executing tool '{tool_name}': Invalid parameters - {str(e)}"
            logger.error(error_msg)
            return error_msg

        except Exception as e:
            # Return a structured error message
            error_msg = f"Error executing tool '{tool_name}': {str(e)}"
            logger.error(error_msg, exc_info=True)
            return error_msg

    def count_tokens_in_response(self, response: Any) -> int:
        """
        Count tokens in the API response content.

        Args:
            response: API response object

        Returns:
            Token count for the response
        """
        try:
            # Extract all text content from response
            content_parts = []
            for content_block in response.content:
                if content_block.type == "text":
                    content_parts.append(content_block.text)
                elif content_block.type == "tool_use":
                    # Include tool use information in token count
                    content_parts.append(json.dumps({
                        "name": content_block.name,
                        "input": content_block.input
                    }))

            # Combine all content and count tokens
            combined_content = "\n".join(content_parts)
            return self.count_tokens_for_content(combined_content)

        except Exception as e:
            logger.warning(f"Error counting response tokens, using fallback: {e}")
            # Fallback estimation
            total_chars = 0
            for content_block in response.content:
                if content_block.type == "text":
                    total_chars += len(content_block.text)
                elif content_block.type == "tool_use":
                    total_chars += len(json.dumps(content_block.input))
            return total_chars // 4

    def get_conversation_stats(self) -> Dict[str, Any]:
        """
        Get current conversation statistics.

        Returns:
            Dictionary containing token usage and other stats
        """
        total_tokens = self.current_conversation_input_tokens + self.current_conversation_output_tokens
        return {
            "input_tokens": self.current_conversation_input_tokens,
            "output_tokens": self.current_conversation_output_tokens,
            "total_tokens": total_tokens,
            "max_tokens": MAX_TOTAL_TOKENS,
            "usage_percentage": (total_tokens / MAX_TOTAL_TOKENS) * 100,
            "messages_count": len(self.session_state.messages),
            "circuit_breaker_state": self.circuit_breaker.get_state_info()
        }

    def reset_circuit_breaker(self) -> None:
        """
        Manually reset the circuit breaker to closed state.

        Use this when you know the service has recovered.
        """
        self.circuit_breaker.state = CircuitState.CLOSED
        self.circuit_breaker.failure_count = 0
        self.circuit_breaker.half_open_calls = 0
        logger.info("Circuit breaker manually reset to CLOSED state")
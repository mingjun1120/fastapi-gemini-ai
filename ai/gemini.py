import os
import logging
from typing import Optional, List, Dict, Any
from google import genai
from google.genai import types, errors
from .base import AIPlatform

class Gemini(AIPlatform):
    def __init__(self, api_key: str, system_prompt: str = None, model_name: str = "gemini-2.0-flash-001", temperature: float = 0.7):
        # Input validation
        if not api_key:
            raise ValueError("API key is required")
        
        self.api_key = api_key
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.tools = [
            types.Tool(googleSearch=types.GoogleSearch())
        ]
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def chat(self, prompt: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Generate a response using the Gemini API
        
        Args:
            prompt: User input prompt
            conversation_history: Optional list of previous messages
            
        Returns:
            Generated response text
            
        Raises:
            ValueError: If prompt is empty
            APIError: If API call fails
        """
        # Input validation
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        
        try:
            # Build contents array - always initialize
            contents = []
            
            # Add conversation history if provided
            if conversation_history:
                for msg in conversation_history:
                    contents.append(types.Content(
                        role=msg.get("role", "user"),
                        parts=[types.Part.from_text(text=msg["content"])]
                    ))
            
            # Add current user prompt
            contents.append(types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)]
            ))
            
            # Build configuration - always initialize
            config_params = {
                "tools": self.tools,
                "temperature": self.temperature,
                "max_output_tokens": 8192,
                "top_p": 0.95
            }
            
            # Add system instruction only if provided
            if self.system_prompt and self.system_prompt.strip():
                config_params["system_instruction"] = self.system_prompt
                config_params["thinking_config"] = types.ThinkingConfig(
                    thinking_budget=-1
                )
            
            generate_content_config = types.GenerateContentConfig(**config_params)
            
            # Make API call with error handling
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=generate_content_config,
            )
            
            # Validate response
            if not response or not hasattr(response, 'text'):
                raise ValueError("Invalid response from API")
            
            return response.text
            
        except errors.APIError as e:
            self.logger.error(f"Gemini API error: {e.code} - {e.message}")
            raise errors.APIError(f"API call failed: {e.message}")
        except Exception as e:
            self.logger.error(f"Unexpected error in chat: {str(e)}")
            raise RuntimeError(f"Chat generation failed: {str(e)}")

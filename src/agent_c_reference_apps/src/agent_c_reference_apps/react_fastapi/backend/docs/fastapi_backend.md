# FastAPI Backend Documentation

## Overview

This FastAPI application serves as a backend for an agent-based system, providing endpoints for session management, file
uploads, persona management, and real-time chat functionality. The application integrates with various AI models and
toolsets to provide a flexible and extensible agent-based communication system.

## Key Components

### Session Management

- Session initialization and verification
- Settings updates for active sessions
- Session cleanup and deletion

### Agent Configuration

- Model selection and configuration
- Tool management and updates
- Persona management

### Communication

- Real-time chat functionality with streaming responses
- File upload capabilities
- Tool registry and management

## API Endpoints

### Session Management

#### `GET /initialize`

Initializes a new session with specified parameters. Note, while we do pass both temperature and reasoning_effort, the
agent's init method will only act on the parameter that is suitable to the model type

Parameters:

- `temperature` (float, default=0.5): Model temperature setting
- `reasoning_effort` (str, default="medium"): Reasoning effort settings
- `model_name` (str, default="gpt-4o"): Name of the AI model
- `backend` (str, default="openai"): Backend service provider
- `persona_name` (str): Optional persona identifier

Returns:

- `session_id`: Unique session identifier
- `agent_c_session_id`: Agent-specific session identifier

#### `GET /verify_session/{session_id}`

Verifies if a session exists and is valid.

Returns:

- `valid`: Boolean indicating session validity

#### `DELETE /sessions`

Deletes all active sessions and cleans up resources.

Returns:

- Status message with number of sessions deleted

### Settings Management

#### `POST /update_settings`

Updates agent settings for a given session.
Note, while we do pass both temperature and reasoning_effort, it will only update the parameter that is suitable to the
model type based on model config
It is controlled by this logic below - model_config is loaded from the model_configs.json file

```python

model_params = {}
if "temperature" in model_config["parameters"]:
    model_params["temperature"] = temperature if temperature is not None else
        model_config["parameters"]["temperature"]["default"]
elif "reasoning_effort" in model_config["parameters"]:
    model_params["reasoning_effort"] = reasoning_effort if reasoning_effort is not None else
        model_config["parameters"]["reasoning_effort"]["default"]
```

Parameters:

- `session_id` (str, required)
- `custom_prompt` (str, optional)
- `temperature` (float, optional)
- `reasoning_effort` (str, optional)
- `persona_name` (str, optional)
- `model_name` (str, optional)
- `backend` (str, optional)

### File Management

**Note** - File uploads are not currently implemented in the frontend and not sent to backend

#### `POST /upload_file`

Handles file uploads and associates them with a session.

Parameters:

- `session_id` (str, required)
- `file` (UploadFile, required)

Returns:

- Upload status and filename

### Persona Management

#### `GET /personas`

Retrieves available personas from the personas directory.

Returns:

- List of personas with name, content, and file information

### Model Management

#### `GET /models`

Retrieves available models from configuration.

Returns:

- List of models with capabilities and parameters

### Tool Management

#### `GET /get_tools`

Retrieves available tools and their configurations.

Returns:

- Essential tools
- Tool groups by category
- Tool name mappings

#### `POST /update_tools`

Updates the active tools for a session.

Parameters:

- `session_id` (str, required)
- `tools` (str, required): JSON string of tool names

#### `GET /get_agent_tools/{session_id}`

Retrieves the tools configured for a specific agent.

Returns:

- List of initialized tools

#### `GET /get_agent_config/{session_id}`

Retrieves the complete configuration for an agent.

Returns:

- Full agent configuration including model info and settings

### Chat Functionality

#### `POST /chat`

Handles chat messages and returns streaming responses.
Note that we only pass custom_prompt to the agent's chat method if it is not None. We do NOT pass in persona. In this UI
we relegate personas to loading them into the model's prompt. This is a design decision that can be changed. It's easier
to
have the front end load personas in a dropdown, when a user selects a persona, that text is loaded into the prompt. Then
if the user edits it (or not) that is the prompt sent to the chat method.

Parameters:

- `session_id` (str, required)
- `message` (str, required)
- `custom_prompt` (str, optional)

Returns:

- Streaming response with agent's reply

## Configuration

### Environment Setup

- `main.py` Has a config directory can be specified via `CONFIG_DIR` environment variable.
  These helper variables are used to locate the model_configs.json file
  **Note** - Personas are loaded from `persona_dir = os.path.join(os.getcwd(), "personas")`, this is why the setup is
  important.
-

```python 
BASE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = os.getenv('CONFIG_DIR', BASE_DIR)
MODEL_CONFIG_PATH = os.path.join(CONFIG_DIR, "model_configs.json")
```

### Model Configuration

- Model configurations stored in `model_configs.json`
- Supports multiple vendors and model types
- Configurable parameters per model
- Top level is vendors, then models, then parameters.  Vendors are `openai` and `claude`. This is because this value is sent to the agent init's `backend` variable and that is hardcoded that way. #TODO: Should change to `open_ai` and `anthropic` to match `tool_vendor` in `tool_set`
- Chat models are ones that do not think/reason. They use temperature as a key parameter
- Reasoning models are ones that do think/reason. They use reasoning_effort as a key parameter of low, medium, high.
```json
{
  "vendors": [
    {
      "vendor": "openai",
      "models": [
        {
          "id": "gpt-4o",
          "model_type": "chat",
          "ui_name": "GPT-4o",
          "description": "Latest GPT-4 OpenAI model for chat interactions.",
          "parameters": {
            "temperature": {
              "default": 0.5,
              "min": 0,
              "max": 2,
              "required": true
            },
            "max_tokens": {
              "default": 4096,
              "required": false
            }
          },
          "capabilities": {
            "supports_tools": true,
            "multi_modal": true
          },
          "allowed_inputs": {
            "image": true,
            "video": false,
            "audio": false,
            "file": false
          }
        }
      ]
    }
  ]
}
```

### CORS Configuration
This needs security updating, but for now.
- Configured to allow all origins (`"*"`)
- Supports all methods and headers
- Credentials allowed



## Dependencies
Required Python packages (see `requirements.txt` and installable via `pyproject.toml`):
- FastAPI
- uvicorn
- pydantic
- It should include core agent_c packages when it's moved outside of the repo


## Security Considerations
- CORS is currently configured to allow all origins
- No built-in authentication/authorization
- File uploads should be properly sanitized in production
- Consider adding rate limiting for production use

## Logging
The application uses Python's built-in logging:

- Logger name: "main"
- Log level: DEBUG
- Logs session creation, tool updates, and errors

## Future Considerations
1. Authentication/Authorization implementation
2. Rate limiting
3. More restrictive CORS policy
4. Enhanced error handling
5. Improved file upload security
6. Performance optimization for large scale deployments
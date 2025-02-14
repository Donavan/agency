import glob
from pathlib import Path
from typing import List, Dict
from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from starlette.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv

from agent_c import Toolset

from agent_c_reference_apps.react_fastapi.backend.backend_app.agent_manager import AgentManager
from agent_c_reference_apps.react_fastapi.backend.util.logging_utils import LoggingManager

# Ensure all our toolsets get registered
from agent_c_tools.tools import *  # noqa
# from agent_c_demo.tools import *  # noqa
# from agent_c_voice.tools import *  # noqa
# from agent_c_rag.tools import *  # noqa

logger = logging.getLogger("main")
logger.setLevel(logging.DEBUG)

BASE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = os.getenv('CONFIG_DIR', BASE_DIR)
MODEL_CONFIG_PATH = os.path.join(CONFIG_DIR, "model_configs.json")

load_dotenv(override=True)

logging_manager = LoggingManager("main")
logger = logging_manager.get_logger()

app = FastAPI(title="Agent C FastAPI Backend")

# CORS setup (adjust origins as necessary):
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the manager
agent_manager = AgentManager()


@app.get("/initialize")
async def initialize_session(
        temperature: float = 0.5,
        model_name: str = "gpt-4o",
        backend: str = "openai",
        reasoning_effort: str = "medium",
        persona_name: str = "",
):
    """
    Creates a fresh session with the provided parameters.
    """
    try:
        # Create a new session with both model and backend parameters
        logging.debug(f"Creating new session with model: {model_name}, backend: {backend}")
        model_params = {"temperature": temperature, "reasoning_effort": reasoning_effort}
        # we're passing both temp/reasoning_effort to create_session, but only one will be used based on the model. That's controlled in teh actual agent's init method.
        new_session_id = await agent_manager.create_session(
            llm_model=model_name,
            backend=backend,
            persona_name=persona_name,
            **model_params
        )

        logger.debug(f"Created new session: {new_session_id} with model: {model_name}, backend: {backend}")
        logger.debug(f"Current sessions: {list(agent_manager.sessions.keys())}")
        logger.debug(f"Agent_C_Session_ID: {agent_manager.get_session_data(new_session_id)}")
        return {"session_id": new_session_id,
                "agent_c_session_id": agent_manager.get_session_data(new_session_id).get('agent_c_session_id',
                                                                                         "Unknown")}

    except Exception as e:
        logger.error(f"Error during session initialization: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/verify_session/{session_id}")
async def verify_session(session_id: str):
    """
    Verifies if a session exists and is valid
    """
    session_data = agent_manager.get_session_data(session_id)
    return {"valid": session_data is not None}


@app.delete("/sessions")
async def delete_all_sessions():
    """
    Delete all active sessions and cleanup their resources.

    Returns:
        dict: Status message with number of sessions deleted

    Raises:
        HTTPException: If there's an error during session cleanup
    """
    try:
        # Get count of sessions before deletion
        session_count = len(agent_manager.sessions)

        # Create list of session IDs to avoid modifying dict during iteration
        session_ids = list(agent_manager.sessions.keys())

        # Clean up each session
        for session_id in session_ids:
            await agent_manager.cleanup_session(session_id)

        logger.info(f"Deleted {session_count} sessions")
        logger.debug(f"Hanging sessions from deletion: {list(agent_manager.sessions.keys())}")

        return {
            "status": "success",
            "message": f"Successfully deleted {session_count} sessions",
            "deleted_count": session_count
        }

    except Exception as e:
        logger.error(f"Error deleting sessions: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete sessions: {str(e)}"
        )


@app.post("/update_settings")
async def update_settings(
        session_id: str = Form(...),
        custom_prompt: str = Form(None),
        temperature: float = Form(None),
        reasoning_effort: str = Form(None),
        persona_name: str = Form(None),
        model_name: str = Form(None),
        backend: str = Form(None)
):
    """
    Update agent settings for a given session.
    Handles both temperature-based and reasoning-based models.
    """
    logger.debug(f"Updating settings for session {session_id} with params: custom prompt - {custom_prompt},\n"
                 f"temp - {temperature},\nreasoning_effort - {reasoning_effort},\n"
                 f"persona- {persona_name},\nmodel - {model_name},\nbackend {backend}")

    session_data = agent_manager.get_session_data(session_id)
    if not session_data:
        return {"error": "Invalid session_id"}

    agent = session_data["agent"]
    updates_made = False

    try:
        # Handle model change first - this will reinitialize the agent
        if model_name is not None and model_name != agent.model_name:
            logger.info(f"Model change requested for session {session_id}: {agent.model_name} -> {model_name}")

            # Load model config to determine correct parameters
            with open(MODEL_CONFIG_PATH, 'r') as f:
                config = json.load(f)

            # Find the model configuration
            model_config = None
            for vendor in config["vendors"]:
                for model in vendor["models"]:
                    if model["id"] == model_name:
                        model_config = model
                        break
                if model_config:
                    break

            if not model_config:
                raise ValueError(f"Model configuration not found for {model_name}")

            # Determine the correct parameter to use
            model_params = {}
            if "temperature" in model_config["parameters"]:
                model_params["temperature"] = temperature if temperature is not None else \
                    model_config["parameters"]["temperature"]["default"]
            elif "reasoning_effort" in model_config["parameters"]:
                model_params["reasoning_effort"] = reasoning_effort if reasoning_effort is not None else \
                    model_config["parameters"]["reasoning_effort"]["default"]

            # Reuse create_session with existing session ID and appropriate parameters
            session_id = await agent_manager.create_session(
                llm_model=model_name,
                backend=backend,
                **model_params,
                persona_name=persona_name if persona_name is not None else agent.persona_name,
                existing_session_id=session_id
            )
            updates_made = True

        else:
            # Update other settings if no model change
            if temperature is not None and hasattr(agent, 'temperature'):
                agent.temperature = temperature
                updates_made = True
            if reasoning_effort is not None and hasattr(agent, 'reasoning_effort'):
                agent.reasoning_effort = reasoning_effort
                updates_made = True
            if custom_prompt is not None:
                agent.custom_persona_text = custom_prompt
                updates_made = True
            if persona_name is not None:
                agent.persona_name = persona_name
                updates_made = True

        if updates_made:
            logger.info(f"Settings updated for session {session_id}")
            return {
                "status": "Settings updated successfully",
                "changes": "Agent reinitialized" if model_name else "Settings modified",
                "agent_c_session_id": agent_manager.get_session_data(session_id).get('agent_c_session_id', "Unknown")
            }
        else:
            logger.info(f"No changes required for session {session_id}")
            return {"status": "No changes required"}

    except Exception as e:
        logger.error(f"Error updating settings for session {session_id}: {str(e)}")
        return {"error": f"Failed to update settings: {str(e)}"}


@app.post("/upload_file")
async def upload_file(
        session_id: str = Form(...),
        file: UploadFile = File(...)
):
    """
    A simple file upload endpoint to associate a file with a user's session.
    For example, store images that might later be used in the conversation.
    """
    session_data = agent_manager.get_session_data(session_id)
    if not session_data:
        return {"error": "Invalid session_id"}

    # For demonstration, read the file in memory or store it on disk, etc.
    # If you have an ImageInput or other approach, you can do that here.
    content = await file.read()
    filename = file.filename
    # Store or process the file as needed...
    logger.info(f"Received file '{filename}' for session {session_id} of size {len(content)} bytes")

    return {"status": "File uploaded successfully", "filename": filename}


@app.get("/personas")
async def get_personas() -> List[Dict[str, str]]:
    """Get list of available personas from personas directory"""
    personas = []
    persona_dir = os.path.join(os.getcwd(), "personas")
    # print(f"Persona directory: {persona_dir}")
    # Ensure directory exists
    if not os.path.isdir(persona_dir):
        logger.warning(f"Persona directory does not exist: {persona_dir}")
        return personas

    # Get all .md files in personas directory
    for file_path in glob.glob(os.path.join(persona_dir, "*.md")):
        name = os.path.basename(file_path)[:-3]  # Remove .md extension

        # Read persona content
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            continue

        personas.append({
            "name": name,
            "content": content,
            "file": os.path.basename(file_path)
        })

    logger.info(f"Found {len(personas)} personas: {[p['name'] for p in personas]}")
    return personas


@app.get("/models")
async def get_models():
    """Get list of available models from model_configs.json"""
    try:
        # Ensure file exists
        if not os.path.isfile(MODEL_CONFIG_PATH):
            logger.warning(f"Model config file does not exist: {MODEL_CONFIG_PATH}")
            return {"models": []}

        # Read and parse model config
        with open(MODEL_CONFIG_PATH, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # Transform the data for frontend consumption
        models = []
        for vendor in config.get("vendors", []):
            vendor_name = vendor.get("vendor")
            for model in vendor.get("models", []):
                models.append({
                    "id": model["id"],
                    "label": model["ui_name"],
                    "description": model["description"],
                    "model_type": model["model_type"],
                    "backend": vendor_name,
                    "parameters": model["parameters"],
                    "capabilities": model["capabilities"],
                    "allowed_inputs": model["allowed_inputs"]
                })

        logger.info(f"Found {len(models)} models in configuration")
        return {"models": models}

    except Exception as e:
        logger.error(f"Error reading model config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/get_tools")
async def get_tools():
    try:
        essential_tools = []
        tool_groups = {
            'Core Tools': [],
            'Demo Tools': [],
            'Voice Tools': [],
            'RAG Tools': []
        }
        tool_name_mapping = {}

        categories = {
            'agent_c_tools': 'Core Tools',
            'agent_c_demo': 'Demo Tools',
            'agent_c_voice': 'Voice Tools',
            'agent_c_rag': 'RAG Tools'
        }

        # Get all tools from the toolsets
        for tool_class in Toolset.tool_registry:
            tool_info = {
                'name': tool_class.__name__,
                'module': tool_class.__module__,
                'doc': tool_class.__doc__,
                'essential': tool_class.__name__ in AgentManager.ESSENTIAL_TOOLS
            }

            if tool_info['essential']:
                essential_tools.append(tool_info)
                continue

            # Categorize non-essential tools
            category = None
            for module_prefix, category_name in categories.items():
                if tool_class.__module__.startswith(module_prefix):
                    category = category_name
                    break

            if category:
                tool_groups[category].append(tool_info)

        # Sort tools
        essential_tools.sort(key=lambda x: x['name'].lower())
        for category in tool_groups:
            tool_groups[category].sort(key=lambda x: x['name'].lower())

        return {
            "essential_tools": essential_tools,
            "groups": tool_groups,
            "categories": list(categories.values()),
            "tool_name_mapping": tool_name_mapping
        }
    except Exception as e:
        logger.error(f"Error retrieving tools: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/update_tools")
async def update_tools(
        session_id: str = Form(...),
        tools: str = Form(...)  # JSON string of tool names
):
    try:
        session_data = agent_manager.get_session_data(session_id)
        if not session_data:
            raise HTTPException(status_code=404, detail="Invalid session ID")

        tool_list = json.loads(tools)
        agent = session_data["agent"]
        await agent.update_tools(tool_list)
        session_data["active_tools"] = tool_list

        return {
            "status": "success",
            "message": "Tools updated successfully",
            "active_tools": tool_list,
            "session_id": session_id,
            "agent_c_session_id": session_data.get('agent_c_session_id', "Unknown")
        }
    except Exception as e:
        logger.error(f"Error updating tools: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/get_agent_tools/{session_id}")
async def get_agent_tools(session_id: str):
    try:
        logger.info(f"get_agent_tools called for session: {session_id}")
        session_data = agent_manager.get_session_data(session_id)
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")

        agent = session_data["agent"]
        config = agent._get_agent_config()
        logger.info(f"Returning tools config: {config['initialized_tools']}")

        return {
            "initialized_tools": config["initialized_tools"],
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error getting agent tools: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/get_agent_config/{session_id}")
async def get_agent_config(session_id: str):
    try:
        logger.info(f"get_agent_config called for session: {session_id}")
        session_data = agent_manager.get_session_data(session_id)
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")

        agent = session_data["agent"]
        config = agent._get_agent_config()

        # Add additional configuration info
        config.update({
            "custom_prompt": agent.custom_persona_text,
            "session_id": session_id,
            "agent_c_session_id": session_data.get("agent_c_session_id"),
            "backend": agent.backend,
            "model_info": {
                "name": agent.model_name,
                "temperature": getattr(agent, "temperature", None),
                "reasoning_effort": getattr(agent, "reasoning_effort", None)
            }
        })

        logger.info(f"Returning agent config: {config}")
        return {
            "config": config,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error getting agent config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def chat_endpoint(
        session_id: str = Form(...),
        message: str = Form(...),
        custom_prompt: str = Form(""),
):
    """
    Endpoint for sending a message and getting a streaming response from the agent.
    We use SSE-like streaming via StreamingResponse.
    """
    logger.info(f"Received chat request for session: {session_id}")
    session_data = agent_manager.get_session_data(session_id)
    logger.debug(f"Available sessions: {list(agent_manager.sessions.keys())}")

    if not session_data:
        logger.error(f"Invalid session ID: {session_id}")
        raise HTTPException(status_code=404, detail="Session not found")

    async def event_stream():
        """Inner generator for streaming response chunks"""
        logger.debug(f"Starting event stream for session: {session_id}")
        try:
            async for token in agent_manager.stream_response(
                    session_id,
                    user_message=message,
                    custom_prompt=custom_prompt,
            ):
                # Each token is a piece of the assistant's reply
                yield token
        except Exception as e:
            logger.error(f"Error in stream_response: {e}")
            yield f"Error: {str(e)}"

    return StreamingResponse(
        event_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

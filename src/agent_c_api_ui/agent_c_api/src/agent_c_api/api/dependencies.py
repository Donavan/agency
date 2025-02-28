from fastapi import Request, Depends, HTTPException
from typing import Dict, Any, Optional, List, Tuple, Set
from pydantic import create_model
import logging
import re
from collections import defaultdict

from agent_c_api.core.agent_manager import AgentManager
from agent_c_api.config.config_loader import get_allowed_params
from agent_c_api.core.util.logging_utils import LoggingManager

logging_manager = LoggingManager(__name__)
logger = logging_manager.get_logger()


def get_agent_manager(request: Request) -> AgentManager:
    return request.app.state.agent_manager


def build_fields_from_config(config: dict) -> dict:
    """
    Recursively build a dictionary of fields for create_model.
    For nested configurations (i.e. those without a top-level "default"),
    create a sub-model.
    """
    fields = {}
    for param, spec in config.items():
        # If spec is a dict but has no "default" key, treat it as nested.
        if isinstance(spec, dict) and "default" not in spec:
            # Extract a required flag if present; assume True if not specified.
            required_flag = spec.get("required", True)
            # Remove keys that aren't actual sub-fields (e.g. "required").
            nested_spec = {k: v for k, v in spec.items() if k != "required"}
            sub_fields = build_fields_from_config(nested_spec)
            # Create a nested model.
            sub_model = create_model(param.capitalize() + "Model", **sub_fields)
            default = ... if required_flag else None
            fields[param] = (sub_model, default)
        else:
            # Flat field: determine the field type based on the parameter name.
            if param == "temperature":
                field_type = float
            elif param in ["max_tokens", "budget_tokens"]:
                field_type = int
            elif param in ["reasoning_effort"]:
                field_type = str
            elif param in ["extended_thinking", "enabled"]:
                field_type = bool
            else:
                field_type = str

            # If spec is a dict, get its "default" value; otherwise, mark as required.
            default = spec.get("default", ...) if isinstance(spec, dict) else ...
            fields[param] = (field_type, default)
    return fields


def analyze_config_structure(config: Dict[str, Any], prefix: str = "", result: Dict[str, Dict] = None) -> Dict[
    str, Dict]:
    """
    Analyze configuration structure to create a mapping of parameter paths.

    Args:
        config: The configuration dictionary
        prefix: Current path prefix for recursive calls
        result: Dictionary to store results

    Returns:
        Dictionary mapping parameter paths to metadata about that path
    """
    if result is None:
        result = {}

    for key, value in config.items():
        current_path = f"{prefix}.{key}" if prefix else key

        # If this is a nested structure (dict without 'default')
        if isinstance(value, dict) and "default" not in value:
            # Mark this as a parent node
            result[current_path] = {
                "is_parent": True,
                "children": [],
                "required": value.get("required", True)
            }

            # Get children but exclude metadata keys like 'required'
            nested_config = {k: v for k, v in value.items() if k != "required"}

            # Recursively process children
            analyze_config_structure(nested_config, current_path, result)

            # Add children to parent's children list
            for child_path in list(result.keys()):
                if child_path.startswith(f"{current_path}.") and "." not in child_path[len(current_path) + 1:]:
                    result[current_path]["children"].append(child_path)
        else:
            # This is a leaf node (actual parameter)
            param_type = None
            if key == "temperature":
                param_type = float
            elif key in ["max_tokens", "budget_tokens"]:
                param_type = int
            elif key in ["reasoning_effort"]:
                param_type = str
            elif key in ["extended_thinking", "enabled"]:
                param_type = bool
            else:
                param_type = str

            result[current_path] = {
                "is_parent": False,
                "type": param_type,
                "default": value.get("default", None) if isinstance(value, dict) else None
            }

    return result


def convert_value_to_type(value: str, target_type: type) -> Any:
    """
    Convert string values from request to appropriate Python types.

    Args:
        value: The value to convert
        target_type: The type to convert to

    Returns:
        Converted value
    """
    if value is None:
        return None

    if target_type == bool:
        return str(value).lower() in ("true", "1", "yes", "y", "on")
    elif target_type == int:
        return int(value)
    elif target_type == float:
        return float(value)
    else:
        return value


def transform_flat_to_nested(params: Dict[str, Any], param_map: Dict[str, Dict]) -> Dict[str, Any]:
    """
    Transform flat parameters to nested structure based on parameter mapping.

    Args:
        params: Dictionary of flat parameters
        param_map: Parameter mapping from analyze_config_structure

    Returns:
        Transformed parameters with nested structure
    """
    result = {}
    processed_params = set()

    # Find all parent nodes in the parameter map
    parent_nodes = {k: v for k, v in param_map.items() if v.get("is_parent", False)}

    # Identify flat parameters that should be nested
    flat_to_nest = {}
    for param in params:
        # Check if this is a flattened nested param (e.g., "extended_thinking.budget_tokens")
        if "." in param:
            parent_param = param.split(".")[0]
            if parent_param in parent_nodes:
                flat_to_nest[param] = param
        # Check if this is a top-level parent param with children
        elif param in parent_nodes:
            # If extended_thinking is boolean in the request but an object in the config
            if parent_nodes[param]["is_parent"]:
                flat_to_nest[param] = param

    # Process simple (non-nested) parameters first
    for param, value in params.items():
        if param not in flat_to_nest and "." not in param:
            # Direct parameter, add to result
            if param in param_map:
                param_type = param_map[param].get("type", str)
                result[param] = convert_value_to_type(value, param_type)
            else:
                # Parameter not in mapping, keep as is
                result[param] = value
            processed_params.add(param)

    # Process nested parameters
    for parent_path, parent_info in parent_nodes.items():
        parent_name = parent_path  # e.g., "extended_thinking"

        # Check if we have any parameters for this parent
        has_child_params = False
        for param in params:
            if param.startswith(f"{parent_name}.") or param == parent_name:
                has_child_params = True
                break

        if has_child_params:
            # Initialize nested object
            nested_obj = {}

            # Process direct children
            for child_path in parent_info.get("children", []):
                child_name = child_path.split(".")[-1]  # e.g., "budget_tokens"

                # Check if we have a flattened version
                flat_key = f"{parent_name}.{child_name}"

                if flat_key in params:
                    # We have a flattened version (e.g., "extended_thinking.budget_tokens")
                    child_type = param_map[child_path].get("type", str)
                    nested_obj[child_name] = convert_value_to_type(params[flat_key], child_type)
                    processed_params.add(flat_key)
                elif child_name in params:
                    # We have the child as a direct param
                    child_type = param_map[child_path].get("type", str)
                    nested_obj[child_name] = convert_value_to_type(params[child_name], child_type)
                    processed_params.add(child_name)

            # Special case: if parent itself is in params (e.g., "extended_thinking=true")
            if parent_name in params:
                # If it's a boolean value but should be an object with "enabled"
                parent_value = params[parent_name]

                # Check if any of the children is named "enabled"
                has_enabled_child = any(
                    path.split(".")[-1] == "enabled" for path in parent_info.get("children", [])
                )

                if has_enabled_child:
                    # If we have an "enabled" child, set it from the parent value
                    nested_obj["enabled"] = convert_value_to_type(parent_value, bool)
                else:
                    # Otherwise, try to extract relevant info from the parent value
                    # This handles cases where the parent might be a JSON string or similar
                    try:
                        if isinstance(parent_value, str) and (
                                parent_value.startswith("{") or parent_value.startswith("[")):
                            import json
                            json_value = json.loads(parent_value)
                            # Merge with existing nested_obj
                            nested_obj.update(json_value)
                    except:
                        # If we can't parse as JSON, treat as a direct value
                        pass

                processed_params.add(parent_name)

            # Add the constructed nested object to the result if not empty
            if nested_obj:
                result[parent_name] = nested_obj

    # Add any remaining parameters that weren't processed
    for param, value in params.items():
        if param not in processed_params:
            result[param] = value

    return result


async def get_dynamic_params(request: Request, model_name: str, backend: str):
    """
    Dynamically validate request parameters based on model configuration.

    Args:
        request: FastAPI request
        model_name: Name of the model to get parameters for
        backend: Backend provider (e.g., 'openai', 'claude')

    Returns:
        Validated parameter object

    Raises:
        HTTPException: If parameters fail validation
    """
    # Look up allowed parameters from the configuration
    allowed_params = get_allowed_params(backend, model_name)
    fields = build_fields_from_config(allowed_params)

    # Analyze the parameter structure
    param_map = analyze_config_structure(allowed_params)

    # Dynamically create a Pydantic model
    DynamicParams = create_model("DynamicParams", **fields)

    try:
        # Convert query parameters to a dict
        params_dict = dict(request.query_params)

        # Transform flat params to nested structure
        transformed_params = transform_flat_to_nested(params_dict, param_map)

        # Validate with the dynamic model
        logger.debug(f"Validating params for {model_name}: {transformed_params}")
        validated_params = DynamicParams.parse_obj(transformed_params)
        return validated_params
    except Exception as e:
        logger.error(f"Parameter validation error for {model_name}: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid parameters: {str(e)}")


async def get_dynamic_form_params(request: Request, agent_manager=Depends(get_agent_manager)):
    """
    Process form parameters with dynamic validation.

    Args:
        request: FastAPI request
        agent_manager: Dependency that provides the agent manager

    Returns:
        Dict containing validated parameters, original form data, and model info
    """
    # Extract form data as a dict
    form = await request.form()
    form_dict = dict(form)

    # Get model_name and backend from form
    model_name = form_dict.get("model_name")
    backend = form_dict.get("backend")

    # If both are provided, validate parameters normally
    if model_name and backend:
        try:
            # Retrieve allowed parameters from the configuration
            allowed_params = get_allowed_params(backend, model_name)

            # Analyze the parameter structure
            param_map = analyze_config_structure(allowed_params)

            # Transform the form data to match expected nested structure
            transformed_form = transform_flat_to_nested(form_dict, param_map)

            # Debug the transformation
            logger.debug(f"Original form: {form_dict}")
            logger.debug(f"Transformed: {transformed_form}")

            fields = build_fields_from_config(allowed_params)
            DynamicFormParams = create_model("DynamicFormParams", **fields)

            try:
                validated_params = DynamicFormParams.parse_obj(transformed_form)
                return {
                    "params": validated_params,
                    "original_form": form_dict,
                    "model_name": model_name,
                    "backend": backend
                }
            except Exception as e:
                logger.error(f"Form parameter validation error: {str(e)}")
                raise HTTPException(status_code=400, detail=f"Invalid parameters: {str(e)}")
        except Exception as e:
            logger.error(f"Error processing form parameters: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Error processing parameters: {str(e)}")

    # If we're missing model info, just return the form data
    # We'll handle fetching the model details in the route handler
    return {
        "params": None,
        "original_form": form_dict,
        "model_name": model_name,
        "backend": backend
    }
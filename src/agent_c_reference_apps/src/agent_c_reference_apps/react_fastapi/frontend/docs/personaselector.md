# PersonaSelector Component Documentation

## Overview
The PersonaSelector component is a React component that provides a user interface for managing AI model settings, persona selection, and custom prompts. It integrates with shadcn/ui components to create a cohesive settings panel.

## Props

| Prop | Type | Description |
|------|------|-------------|
| persona | string | The currently selected persona identifier |
| personas | Array<{name: string, content: string}> | Array of available personas with their configuration |
| customPrompt | string | Custom instructions for the AI model |
| modelName | string | Current model identifier |
| modelConfigs | Array<ModelConfig> | Available model configurations |
| sessionId | string | Current session identifier |
| modelParameters | Object | Current model parameters |
| selectedModel | ModelConfig | Currently selected model configuration |
| onUpdateSettings | function | Callback for settings updates |
| isInitialized | boolean | Flag indicating if the component is initialized |

## State Management

The component maintains local UI state using React's useState hook:
- `error`: Tracks error states during operations
- `selectedPersona`: Manages the currently selected persona in the UI
- `localCustomPrompt`: Manages the custom prompt text in the UI

## Key Features

### 1. Persona Selection
- Allows users to select from predefined personas
- Updates both local state and parent component via callbacks
- Automatically updates custom prompt based on selected persona

### 2. Custom Prompt Management
- Provides a textarea for custom instruction input
- Implements debounced updates to prevent excessive API calls
- Maintains local state for smooth UI experience

### 3. Model Selection
- Grouped display of models by vendor
- Tooltips with model descriptions and details
- Handles model switching with appropriate parameter updates

### 4. Parameter Controls
- Integrates with ModelParameterControls component
- Supports real-time parameter adjustments
- Maintains consistency with selected model capabilities

## Event Handlers

### handlePersonaChange
Manages persona selection changes:
- Updates local state
- Triggers parent component updates
- Updates custom prompt based on selected persona

### handleCustomPromptChange
Handles custom prompt modifications:
- Updates local state immediately for responsive UI
- Defers parent component updates until blur

### handleCustomPromptBlur
Manages custom prompt completion:
- Triggers parent updates only on actual changes
- Prevents unnecessary API calls

### handleParameterChange
Handles model parameter updates:
- Propagates parameter changes to parent component
- Maintains parameter consistency

### handleModelChange
Manages model selection:
- Updates model configuration
- Handles backend changes
- Triggers appropriate parent component updates

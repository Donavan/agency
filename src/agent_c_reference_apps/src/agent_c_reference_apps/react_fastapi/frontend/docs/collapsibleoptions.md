# CollapsibleOptions Component Documentation

## Overview
CollapsibleOptions is a React component that provides a collapsible options panel interface. 
It combines persona selection and tool selection functionality in an expandable/collapsible card format.

## Component Structure
The component is built using the shadcn/ui library components and consists of:
- A collapsible card container
- A header with toggle button
- Two main sections:
  - PersonaSelector
  - ToolSelector

## Props

### UI Control Props
- `isOpen: boolean` - Controls the expanded/collapsed state of the options panel
- `setIsOpen: (boolean) => void` - Callback to update the panel's open state

### Persona Configuration Props
- `persona: string` - Currently selected persona name
- `customPrompt: string` - Custom prompt text if being used
- `temperature: number` - Temperature setting for the model
- `modelName: string` - Name of the current model
- `modelConfigs: object` - Configuration options for available models
- `modelParameters: object` - Current model parameters
- `selectedModel: string` - Currently selected model identifier
- `personas: array` - Available personas to choose from

### Tool Configuration Props
- `availableTools: array` - List of tools that can be equipped
- `activeTools: array` - Currently equipped tools
- `onEquipTools: (tools: array) => void` - Callback when tools are equipped/unequipped

### Session Management Props
- `sessionId: string` - Current session identifier
- `isReady: boolean` - Indicates if the component is ready for interaction
- `isInitialized: boolean` - Indicates if the component has completed initialization

### Callback Props
- `onUpdateSettings: (settings: object) => void` - Callback for when settings are updated

## Sub-Components
### PersonaSelector
Handles the selection and configuration of personas, including:
- Persona selection
- Custom prompt configuration
- Model parameter adjustments

### ToolSelector
Manages the available tools and their activation status:
- Tool selection interface
- Active tool management
- Tool equipping/unequipping functionality

## Styling
The component uses Tailwind CSS classes for styling:
- Uses a semi-transparent white background with backdrop blur
- Implements a shadow for depth
- Maintains consistent padding and spacing
- Responsive design considerations
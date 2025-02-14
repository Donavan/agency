# React Chat Application Documentation

## Architecture Overview

This React application implements a sophisticated chat interface with AI capabilities, tool integration, and dynamic configuration options. The application is built using modern React practices and follows a component-based architecture.

## Core Components

### 1. ChatInterface
**Purpose**: Main chat interface handling message display and interaction
**Key Features**:
- Real-time message streaming
- File upload capabilities
- Tool call integration
- Token usage tracking
- Markdown message rendering

**Data Flow**:
- Manages message state using `useState`
- Handles file uploads via FormData
- Processes streaming responses from API
- Updates tool calls in real-time

### 2. CollapsibleOptions
**Purpose**: Configurable options panel for persona and tool selection
**Components**:
- PersonaSelector: Manages AI persona configuration
- ToolSelector: Handles tool selection and management
- ModelParameterControls: Controls model-specific parameters

### 3. StatusBar
**Purpose**: Displays system status and configuration
**Features**:
- Real-time status updates
- Agent configuration display
- Session management
- Active tools display

## Component Relationships

### Message Flow
1. User input is captured in ChatInterface
2. Messages are processed through the API
3. Responses are streamed and parsed
4. Different message types are rendered:
   - Text (MarkdownMessage)
   - Media (MediaMessage)
   - Tool Calls (ToolCallDisplay)

### Configuration Flow
1. CollapsibleOptions manages settings
2. Changes propagate through onUpdateSettings
3. StatusBar reflects current configuration
4. AgentConfigDisplay shows active settings

## Key Features
### Tool Integration
- Dynamic tool loading via ToolSelector
- Real-time tool execution tracking
- Tool results display in chat
- Tool categorization and management

### Message Rendering
- Markdown support with syntax highlighting
- Media message handling (SVG, HTML, images)
- Token usage tracking
- Structured message organization

### Configuration Management
- Model parameter controls
- Persona selection and customization
- Tool selection and management
- Session state persistence

## Technical Implementation
### State Management
- Local state for UI components
- Prop drilling for configuration
- Session-based persistence
- Real-time status updates

### API Integration
- Streaming response handling
- File upload management
- Tool execution
- Configuration updates

### UI Components
The application uses several shared UI components:
- Button: Common button component
- Card: Container component
- Input: Form input component
- Select: Dropdown selection component
- Toast: Notification component

## Styling
- Tailwind CSS for styling
- Consistent design system
- Responsive layout
- Backdrop blur effects



## Setup and Configuration

### Required Environment Variables
- VITE_API_URL: API endpoint configuration

### Component Dependencies
- React
- Lucide Icons
- Tailwind CSS
- ShadcnUI Components
- React Markdown

## Usage Guidelines
### Adding New Features
1. Follow existing component patterns
2. Maintain consistent styling
3. Update documentation
4. Test thoroughly

### Modifying Existing Components
1. Understand component relationships
2. Test changes thoroughly
3. Update documentation
4. Maintain consistent patterns
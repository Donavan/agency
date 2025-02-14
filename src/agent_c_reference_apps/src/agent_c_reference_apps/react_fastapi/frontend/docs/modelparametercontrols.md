# ModelParameterControls Component Documentation

## Overview
The ModelParameterControls component is a React component that provides an interface for adjusting AI model parameters such as temperature and reasoning effort. It uses shadcn/ui components to create an intuitive parameter adjustment interface with real-time visual feedback.

## Props

| Prop | Type | Description |
|------|------|-------------|
| selectedModel | ModelConfig | The currently selected model configuration including parameter settings |
| onParameterChange | function | Callback function for parameter value changes |
| currentParameters | Object | Current parameter values (optional, defaults to empty object) |

## Parameter Types

### Temperature Control
- Range-based parameter (slider)
- Configurable min/max/step values
- Real-time visual feedback
- Commits changes only when user finishes adjusting

### Reasoning Effort Control
- Discrete options (select)
- Typically "low", "medium", "high"
- Supports custom option sets from model configuration

## State Management

The component maintains three state variables:
- `temperature`: The committed temperature value
- `localTemperature`: The current slider position for smooth UI updates
- `reasoningEffort`: The selected reasoning effort level

## Key Features

### 1. Temperature Control
- Smooth slider interaction with real-time feedback
- Visual markers for different creativity levels
- Separate state for UI updates and committed values
- Informative labels showing current value
- Visual scale from "Focused" to "Creative"

### 2. Reasoning Effort Control
- Dynamic option generation based on model configuration
- Flexible configuration support
- Capitalized display values
- Styled select interface with backdrop blur effects

### 3. Responsive Design
- Clean layout with appropriate spacing
- Visual feedback for user interactions
- Tooltip-style display of current temperature
- Informative helper text

## Methods

### handleTemperatureChange
- Manages real-time temperature slider updates
- Updates local state for smooth UI
- Doesn't trigger backend updates

### handleTemperatureCommit
- Triggers when user finalizes temperature selection
- Updates both local state and parent component
- Calls onParameterChange callback

### handleReasoningEffortChange
- Manages reasoning effort selection changes
- Immediately updates both local state and parent component
- Capitalizes display values

### getReasoningEffortOptions
- Extracts reasoning effort options from model configuration
- Handles multiple configuration formats
- Provides fallback default options

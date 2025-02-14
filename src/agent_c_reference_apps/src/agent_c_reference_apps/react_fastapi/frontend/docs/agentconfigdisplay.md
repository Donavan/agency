# AgentConfigDisplay Component Documentation

## Overview
The AgentConfigDisplay component is a React component that fetches and displays agent configuration information in a tooltip interface. It provides a visual indicator of the current configuration state and detailed information on hover.

## Props

| Prop | Type | Description |
|------|------|-------------|
| sessionId | string | The current session identifier for fetching configuration |
| className | string | Optional CSS classes to apply to the component (defaults to "") |
| settingsVersion | number | Version identifier that triggers configuration refresh |

## State Management

The component maintains two state variables:
- `config`: Stores the fetched configuration data
- `error`: Tracks error states during API calls

## Key Features

### 1. Configuration Fetching
- Automatically fetches configuration when sessionId or settingsVersion changes
- Implements cache control headers to ensure fresh data
- Handles error states gracefully
- Logs configuration changes for debugging

### 2. Visual Display
- Compact icon-based display in normal state
- Expandable tooltip with detailed configuration
- Loading state indicator
- Error handling with graceful fallback

### 3. Configuration Information
Displays the following configuration details:
- Model name
- Backend system
- Temperature (if applicable)
- Reasoning effort (if applicable)
- Persona name
- Number of active tools
- Session ID

## API Integration

### Endpoint
```javascript
GET ${API_URL}/get_agent_config/${sessionId}
```

### Headers
```javascript
{
  'Cache-Control': 'no-cache',
  'Pragma': 'no-cache'
}
```

### Response Structure
```typescript
interface ConfigResponse {
  config: {
    model_info: {
      name: string;
      temperature?: number;
      reasoning_effort?: string;
    };
    backend: string;
    persona_name: string;
    initialized_tools: string[];
    session_id: string;
  }
}
```

## Component States

### Loading State
```jsx
<div className="inline-flex items-center cursor-help">
  <Settings className="w-4 h-4 text-gray-400" />
  <span className="ml-1 text-sm text-gray-400">Loading...</span>
</div>
```

### Error State
- Returns null to prevent display of invalid data
- Logs error to console for debugging

### Active State
- Displays interactive tooltip with configuration details
- Uses Radix UI Portal for proper stacking context
- Implements hover interactions

## Usage Example

```jsx
<AgentConfigDisplay
  sessionId="session-123"
  className="my-2"
  settingsVersion={1}
/>
```

## Implementation Details

### Configuration Change Detection
```javascript
if (config) {
  const changes = {};
  Object.keys(data.config).forEach(key => {
    if (JSON.stringify(data.config[key]) !== JSON.stringify(config[key])) {
      changes[key] = {
        from: config[key],
        to: data.config[key]
      };
    }
  });
  console.log('Config changes:', changes);
}
```

### Configuration Display Formatting
```javascript
const configDisplay = {
  "Model": config.model_info?.name,
  "Backend": config.backend,
  ...(config.model_info?.temperature !== undefined && {
    "Temperature": config.model_info?.temperature.toFixed(2)
  }),
  // ... additional fields
};
```

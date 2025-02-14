# ToolSelector Component Documentation

## Overview
The ToolSelector component is a complex React component that provides an interface for selecting and managing tools in an agent-based system. It consists of three main components: the main ToolSelector, EssentialTools, and ToolCategory.

## Component Architecture

### 1. Main ToolSelector Component
The primary container component that manages tool selection and equipment.

#### Props
| Prop | Type | Description |
|------|------|-------------|
| availableTools | Object | Configuration object containing tool categories and groups |
| onEquipTools | Function | Callback function when tools are equipped |
| activeTools | Array | Currently active tools (defaults to []) |
| sessionId | String | Current session identifier |
| isReady | Boolean | Flag indicating if the agent is ready |

#### State Management
- `selectedTools`: Set of currently selected tool names
- `error`: Error state management
- `toast`: Toast notification state
- `isLoading`: Loading state during tool equipment

### 2. EssentialTools Component
Displays non-toggleable essential tools.

#### Props
| Prop | Type | Description |
|------|------|-------------|
| tools | Array | Array of essential tool objects |

#### Tool Object Structure
```typescript
interface Tool {
  name: string;
  doc?: string;  // Optional description
}
```

### 3. ToolCategory Component
Displays a category of toggleable tools.

#### Props
| Prop | Type | Description |
|------|------|-------------|
| title | String | Category title |
| tools | Array | Array of tools in the category |
| selectedTools | Set | Currently selected tools |
| activeTools | Array | Currently active tools |
| onToolToggle | Function | Callback for tool selection changes |

## Features

### 1. Tool Selection
- Interactive checkboxes for tool selection
- Visual feedback for active tools
- Category-based organization
- Tooltips with tool descriptions

### 2. State Management
- Tracks selected and active tools
- Manages loading states
- Handles errors with toast notifications
- Synchronizes with backend through onEquipTools callback

### 3. UI/UX Features
- Responsive grid layout
- Scrollable tool list
- Loading indicators
- Error messaging
- Toast notifications
- Tool status badges

## Implementation Details

### Tool Selection Logic
```javascript
const handleToolToggle = (toolName) => {
  if (!isReady) {
    showToast('Please wait for agent initialization', 'error');
    return;
  }
  setSelectedTools((prev) => {
    const newSelected = new Set(prev);
    if (newSelected.has(toolName)) {
      newSelected.delete(toolName);
    } else {
      newSelected.add(toolName);
    }
    return newSelected;
  });
};
```

### Tool Equipment Process
```javascript
const handleEquipTools = async () => {
  if (!isReady) {
    showToast('Please wait for agent initialization', 'error');
    return;
  }
  setIsLoading(true);
  try {
    const toolsToEquip = Array.from(selectedTools);
    await onEquipTools(toolsToEquip);
    showToast('Tools equipped successfully!', 'success');
  } catch (error) {
    console.error('Error equipping tools:', error);
    setError('Failed to equip tools: ' + error.message);
    showToast('Failed to equip tools', 'error');
  } finally {
    setIsLoading(false);
  }
};
```

## Styling

### Essential Tools
- Blue theme with badges
- Hover effects on badges
- Visual indicators for essential status

### Category Tools
- Grid layout with responsive columns
- Checkbox-based selection
- Active state highlighting
- Status badges for active tools

### Container Styling
- Card-based layout
- Scrollable content area
- Full-width action button
- Loading state indicators


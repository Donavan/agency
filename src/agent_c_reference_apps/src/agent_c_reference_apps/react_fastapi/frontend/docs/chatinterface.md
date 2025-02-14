# ChatInterface Component Documentation

## Overview
The ChatInterface component is a React-based chat interface that handles real-time message streaming, file uploads, and 
various types of message displays including text, media, and tool calls. It provides a complete chat experience with support for 
markdown rendering, token usage tracking, and multiple message types.

## Props

| Prop | Type | Description |
|------|------|-------------|
| sessionId | string | Unique identifier for the chat session |
| customPrompt | string | Optional custom prompt to be used for the chat |
| modelName | string | Name of the language model to be used |
| modelParameters | object | Configuration parameters for the model |
| onProcessingStatus | function | Callback function for streaming status updates |

## State Management
The component manages several pieces of state:
- `messages`: Array of all chat messages with their types and content
- `inputText`: Current text in the input field
- `isStreaming`: Boolean flag for active message streaming
- `activeToolCalls`: Map of currently active tool calls

## Message Types
The component handles multiple message types:
1. User Messages (`role: "user"`)
   - Plain text messages sent by the user
   
2. Assistant Messages (`role: "assistant"`)
   - Text responses with markdown support
   - Include token usage statistics when available
   
3. Tool Calls
   - Special messages that represent tool operations
   - Can include multiple tool calls in a single message
   
4. Media Messages
   - Support for various media types
   - Rendered through the MediaMessage component
   
5. System Messages
   - Internal system notifications and status updates

## Key Features
### File Upload
- NOT CURRENTLY IMPLEMENTED

### Message Streaming
- Real-time message streaming with SSE (Server-Sent Events)
- Supports partial message updates
- Handles multiple message types in the stream

### Tool Call Management
- Tracks tool calls through their lifecycle
- Supports multiple simultaneous tool calls
- Updates tool call status and results in real-time

### UI/UX Features
- Automatic scroll to bottom on new messages
- Keyboard shortcuts (Enter to send)
- Loading states and disabled inputs during streaming
- Responsive design with proper spacing and layout
- Support for markdown rendering in messages

## Key Methods
### Message Handling
```javascript
handleSendMessage()
- Processes and sends user messages
- Initiates streaming response
- Handles errors and updates UI state

processLine(line)
- Processes individual lines from the message stream
- Parses different message types
- Updates relevant state based on message content

handleToolStart(toolDetails)
- Initializes new tool calls
- Updates UI with tool call status

handleToolEnd(toolResults)
- Processes tool call results
- Updates UI with completion status
```

### File Management
```javascript
handleUploadFile()
- Processes file uploads
- Sends files to backend
- Updates UI with upload status
```

## Styling
The component uses Tailwind CSS for styling with:
- Responsive design principles
- Glass morphism effects (backdrop-blur)
- Consistent color scheme
- Proper spacing and padding
- Rounded corners and shadows for depth

## Error Handling
The component includes error handling for:
- Failed message sends
- File upload errors
- Stream processing errors
- Invalid message formats
- Tool call failures

## Future Code Improvements
1. Add TypeScript interfaces for better type safety
2. Add prop validation using PropTypes
3. Add missing JSDoc comments for key methods
4. Consider breaking down into smaller subcomponents for better maintainability
5. Use of useRef for scroll management
6. Efficient message update handling 
7. Proper cleanup of resources 
8. Stream handling for large messages 
9. Proper error boundaries and fallbacks
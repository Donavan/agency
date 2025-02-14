import React, {useState, useRef, useEffect} from "react";
import {Card} from "@/components/ui/card";
import {Button} from "@/components/ui/button";
import {Input} from "@/components/ui/input";
import {Send, Upload, Brain, User} from "lucide-react";
import {ScrollArea} from "@/components/ui/scroll-area";
import ToolCallDisplay from "./ToolCallDisplay";
import MediaMessage from './MediaMessage';
import TokenUsageDisplay from './TokenUsageDisplay';
import MarkdownMessage from './MarkdownMessage';

const API_URL = import.meta.env.VITE_API_URL;

/**
 * ChatInterface component provides a complete chat interface with support for
 * message streaming, file uploads, and various message types including text,
 * media, and tool calls.
 *
 * @component
 * @param {Object} props
 * @param {string} props.sessionId - Unique identifier for the chat session
 * @param {string} [props.customPrompt] - Optional custom prompt for the chat
 * @param {string} props.modelName - Name of the language model to use
 * @param {Object} props.modelParameters - Configuration parameters for the model
 * @param {Function} props.onProcessingStatus - Callback for streaming status updates
 */
const ChatInterface = ({sessionId, customPrompt, modelName, modelParameters, onProcessingStatus}) => {
    const [messages, setMessages] = useState([]);
    const [inputText, setInputText] = useState("");
    const [isStreaming, setIsStreaming] = useState(false);
    const [activeToolCalls, setActiveToolCalls] = useState(new Map()); // Track active tool calls
    const fileInputRef = useRef(null);
    const messagesEndRef = useRef(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({behavior: "smooth"});
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    useEffect(() => {
        // console.log('Streaming state:', isStreaming);
        if (onProcessingStatus) {
            onProcessingStatus(isStreaming);
        }
    }, [isStreaming, onProcessingStatus]);

    /**
     * Handles file upload to the server
     * @returns {Promise<void>}
     * @throws {Error} If the file upload fails
     */
    const handleUploadFile = async () => {
        if (!fileInputRef.current?.files?.length) return;

        const formData = new FormData();
        formData.append("session_id", sessionId);
        formData.append("file", fileInputRef.current.files[0]);

        try {
            const response = await fetch(`${API_URL}/upload_file`, {
                method: "POST",
                body: formData,
            });
            const data = await response.json();
            setMessages((prev) => [
                ...prev,
                {
                    role: "system",
                    type: "content",
                    content: `File uploaded: ${fileInputRef.current.files[0].name}`,
                },
            ]);
        } catch (error) {
            console.error("Error uploading file:", error);
            setMessages((prev) => [
                ...prev,
                {
                    role: "system",
                    type: "content",
                    content: "Error uploading file",
                },
            ]);
        }
    };

    /**
     * Handles the start of a tool call operation
     * @param {Array<Object>} toolDetails - Array of tool call details
     * @param {string} toolDetails[].id - Unique identifier for the tool call
     * @param {string} toolDetails[].name - Name of the tool being called
     * @param {Object} toolDetails[].arguments - Arguments for the tool call
     */
    const handleToolStart = (toolDetails) => {
        console.log("=== handleToolStart called ===");
        console.log("Received tool details:", toolDetails);

        // Process each tool call in the array
        const newToolCalls = toolDetails.map((tool) => ({
            id: tool.id || tool.tool_call_id,
            name: tool.name || tool.function?.name,  // Handle function name if in nested structure
            arguments: tool.arguments || tool.function?.arguments,  // Handle nested arguments
            results: null
        }));

        // Update state with the new tool calls
        setActiveToolCalls((prev) => {
            const newMap = new Map(prev);
            newToolCalls.forEach((call) => newMap.set(call.id, call));
            return newMap;
        });

        setMessages((prev) => {
            const lastMessage = prev[prev.length - 1];
            if (lastMessage?.type === "tool_calls") {
                // Update existing tool calls message
                return prev.map((msg, i) =>
                    i === prev.length - 1
                        ? {...msg, toolCalls: [...(msg.toolCalls || []), ...newToolCalls]}
                        : msg
                );
            }
            // Create new tool calls message
            return [...prev, {
                role: "assistant",
                type: "tool_calls",
                toolCalls: newToolCalls
            }];
        });
    };

    /**
     * Handles the completion of a tool call operation and updates relevant state
     * @param {Object} toolResults - Results from the completed tool operation
     * @param {string} toolResults.tool_call_id - Unique identifier of the completed tool call
     * @param {string|Object} toolResults.content - Results/output from the tool operation
     * @throws {Warning} If required fields are missing from tool results
     */
    const handleToolEnd = (toolResults) => {
        console.log("=== handleToolEnd called ===");
        console.log("Received tool results:", toolResults);

        if (!toolResults) {
            console.warn("Invalid tool results:", toolResults);
            return;
        }

        const toolCallId = toolResults.tool_call_id;
        const content = toolResults.content;

        if (!toolCallId || !content) {
            console.warn("Missing required tool result fields:", toolResults);
            return;
        }

        setActiveToolCalls((prev) => {
            const updated = new Map(prev);
            if (updated.has(toolCallId)) {
                const call = updated.get(toolCallId);
                updated.set(toolCallId, {...call, results: content});
            }
            return updated;
        });

        setMessages((prev) => {
            return prev.map((message) => {
                if (message.type === "tool_calls") {
                    const updatedToolCalls = message.toolCalls.map((call) => {
                        if (call.id === toolCallId) {
                            return {...call, results: content};
                        }
                        return call;
                    });
                    return {...message, toolCalls: updatedToolCalls};
                }
                return message;
            });
        });
    };

    /**
     * Sends a message to the chat backend and processes the response stream
     * @returns {Promise<void>}
     * @throws {Error} If the message send fails or stream processing encounters an error
     */
    const handleSendMessage = async () => {
        if (!inputText.trim() || isStreaming) return;

        try {
            setIsStreaming(true);
            setActiveToolCalls(new Map());

            // Add user message
            setMessages((prev) => [
                ...prev,
                {role: "user", type: "content", content: inputText},
            ]);

            const userText = inputText;
            setInputText("");

            const formData = new FormData();
            formData.append("session_id", sessionId);
            formData.append("message", userText);
            formData.append("custom_prompt", customPrompt || "");

            // Add the correct parameter based on model type
            if (modelParameters.temperature !== undefined) {
                formData.append("temperature", modelParameters.temperature);
            }
            if (modelParameters.reasoning_effort !== undefined) {
                formData.append("reasoning_effort", modelParameters.reasoning_effort);
            }

            formData.append("llm_model", modelName);

            const response = await fetch(`${API_URL}/chat`, {
                method: "POST",
                body: formData,
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            if (!response.body) {
                throw new Error("No response body");
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder("utf-8");
            let buffer = "";

            try {
                while (true) {
                    const {value, done} = await reader.read();

                    if (done) {
                        // Handle any remaining buffer content
                        if (buffer.trim()) {
                            await processLine(buffer.trim());
                        }
                        break;
                    }

                    buffer += decoder.decode(value, {stream: true});

                    // Process complete lines
                    let newlineIndex;
                    while ((newlineIndex = buffer.indexOf("\n")) !== -1) {
                        const line = buffer.slice(0, newlineIndex).trim();
                        buffer = buffer.slice(newlineIndex + 1);

                        if (line) {
                            await processLine(line);
                        }
                    }
                }
            } finally {
                reader.releaseLock();
            }
        } catch (error) {
            console.error("Error in chat:", error);
            setMessages((prev) => [
                ...prev,
                {
                    role: "system",
                    type: "content",
                    content: `Error: ${error.message}`,
                },
            ]);
        } finally {
            setIsStreaming(false);
        }
    };

// Helper function to process each line of the stream
    /**
     * Processes a single line from the message stream
     * @param {string} line - Raw line from the message stream
     * @returns {Promise<void>}
     * @throws {Error} If the line cannot be parsed or processed
     */
    const processLine = async (line) => {
        // Check if the line is the termination marker.
        if (line === "null") {
            console.log("Termination marker received. Ending stream processing.");
            return;
        }
        try {
            const parsed = JSON.parse(line);
            // If parsed is null (in case JSON.parse("null") returns null), exit early.
            if (parsed === null) {
                console.log("Termination marker received after parsing.");
                return;
            }
            console.log("=== Received message ===");
            console.log("Full parsed data:", parsed);

            switch (parsed.type) {
                case "content":
                    setMessages((prev) => {
                        const last = prev[prev.length - 1];
                        if (last?.role === "assistant" && last?.type === "content") {
                            return prev.map((msg, i) =>
                                i === prev.length - 1
                                    ? {...msg, content: msg.content + parsed.data}
                                    : msg
                            );
                        }
                        return [...prev, {role: "assistant", type: "content", content: parsed.data}];
                    });
                    break;

                case "tool_calls":
                    if (parsed.tool_calls) {
                        handleToolStart(parsed.tool_calls);
                    }
                    break;

                case "tool_results":
                    if (parsed.tool_results) {
                        parsed.tool_results.forEach((result) => handleToolEnd(result));
                    }
                    break;

                case "render_media":
                    setMessages((prev) => [
                        ...prev,
                        {
                            role: "assistant",
                            type: "media",
                            content: parsed.content,
                            contentType: parsed.content_type,
                            metadata: parsed.metadata
                        },
                    ]);
                    break;

                case "completion_status":
                    if (!parsed.data.running) {
                        if (parsed.data.input_tokens || parsed.data.output_tokens) {
                            setMessages((prev) => {
                                const lastAssistantMessage = [...prev].reverse().find(
                                    (msg) => msg.role === "assistant" && msg.type === "content"
                                );
                                if (!lastAssistantMessage) return prev;
                                return prev.map((msg) => {
                                    if (msg === lastAssistantMessage) {
                                        return {
                                            ...msg,
                                            tokenUsage: {
                                                prompt_tokens: parsed.data.input_tokens,
                                                completion_tokens: parsed.data.output_tokens,
                                                total_tokens: parsed.data.input_tokens + parsed.data.output_tokens,
                                            },
                                        };
                                    }
                                    return msg;
                                });
                            });
                        }
                    }
                    break;

                // These cases don’t require any specific handling
                case "interaction_start":
                case "interaction_end":
                case "history":
                    break;

                default:
                    console.warn("Unknown message type:", parsed.type);
            }
        } catch (err) {
            console.error("Error processing message:", err);
            throw err;
        }
    };

    const handleKeyPress = (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            handleSendMessage();
        }
    };

    return (
        <Card className="flex flex-col h-full bg-white/50 backdrop-blur-sm border shadow-lg rounded-xl relative z-0">
            <ScrollArea className="flex-1 px-4 py-3">
                <div className="space-y-4">
                    {messages.map((msg, idx) => {
                        // === 1) USER MESSAGES ===
                        if (msg.role === "user") {
                            return (
                                <div key={idx} className="flex justify-end items-start gap-2">
                                    <div
                                        className="max-w-[80%] rounded-2xl px-4 py-2 shadow-sm bg-blue-500 text-white ml-12 rounded-br-sm">
                                        {msg.content}
                                    </div>
                                    <User className="h-6 w-6 text-blue-500"/>
                                </div>
                            );
                        }

                        // === 2) ASSISTANT TEXT ===
                        if (msg.role === "assistant" && msg.type === "content") {
                            return (
                                <div key={idx} className="flex flex-col">
                                    <div className="flex justify-start items-start gap-2">
                                        <Brain className="h-6 w-6 text-purple-500"/>
                                        <div
                                            className="max-w-[80%] rounded-2xl px-4 py-2 shadow-sm bg-purple-50 text-purple-800 mr-12 rounded-bl-sm">
                                            <MarkdownMessage content={msg.content}/>
                                            {msg.tokenUsage && <TokenUsageDisplay usage={msg.tokenUsage}/>}
                                        </div>
                                    </div>
                                </div>
                            );
                        }

                        // === 3) TOOL CALLS (single container for all tool calls) ===
                        if (msg.type === "tool_calls") {
                            return (
                                <div key={idx}>
                                    <ToolCallDisplay toolCalls={msg.toolCalls}/>
                                </div>
                            );
                        }

                        // === 4) MEDIA ===
                        if (msg.type === "media") {
                            return (
                                <div key={idx}>
                                    <MediaMessage message={msg}/>
                                </div>
                            );
                        }

                        // === 5) SYSTEM MESSAGES ===
                        if (msg.role === "system") {
                            return (
                                <div key={idx} className="flex justify-start">
                                    <div
                                        className="max-w-[80%] rounded-2xl px-4 py-2 shadow-sm bg-gray-100 text-gray-600">
                                        {msg.content}
                                    </div>
                                </div>
                            );
                        }

                        return null;
                    })}
                    <div ref={messagesEndRef}/>
                </div>
            </ScrollArea>

            {/* Footer with file upload and message input */}
            <div className="border-t bg-white/50 backdrop-blur-sm p-4 space-y-3 rounded-b-xl">
                <div className="flex gap-2">
                    <Input
                        type="file"
                        ref={fileInputRef}
                        className="block w-full text-sm text-gray-500
              file:mr-4 file:py-2 file:px-4
              file:rounded-xl file:border-0
              file:text-sm file:font-semibold
              file:bg-blue-50 file:text-blue-700
              hover:file:bg-blue-100
              focus:outline-none focus:ring-2 focus:ring-blue-200 focus:ring-opacity-50
              rounded-xl
              cursor-pointer
              transition-all
              border border-gray-200
              bg-white/50 backdrop-blur-sm"
                    />
                    <Button
                        onClick={handleUploadFile}
                        variant="outline"
                        size="icon"
                        className="shrink-0 rounded-xl border-gray-200 bg-white/50 backdrop-blur-sm hover:bg-white/80 transition-colors"
                    >
                        <Upload className="h-4 w-4"/>
                    </Button>
                </div>

                <div className="flex gap-2">
                    <Input
                        type="text"
                        placeholder="Type your message..."
                        value={inputText}
                        onChange={(e) => setInputText(e.target.value)}
                        onKeyPress={handleKeyPress}
                        disabled={isStreaming}
                        className="flex-1 rounded-xl border-gray-200 bg-white/50 backdrop-blur-sm transition-colors
              hover:bg-white/80 focus:border-blue-300 focus:ring focus:ring-blue-200 focus:ring-opacity-50
              placeholder-gray-400"
                    />
                    <Button
                        onClick={handleSendMessage}
                        disabled={isStreaming}
                        size="icon"
                        className="shrink-0 rounded-xl bg-blue-500 hover:bg-blue-600 transition-colors"
                    >
                        <Send className="h-4 w-4"/>
                    </Button>
                </div>
            </div>
        </Card>
    );
};

export default ChatInterface;
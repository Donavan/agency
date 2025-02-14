import React, {useState, useEffect, useRef} from "react";
import {Alert, AlertDescription} from "@/components/ui/alert";
import ChatInterface from "./components/ChatInterface";
import StatusBar from "./components/StatusBar";
import CollapsibleOptions from "./components/CollapsibleOptions";

const API_URL = import.meta.env.VITE_API_URL;

/**
 * Root component for the Agent C Conversational Interface.
 * Manages application state, chat sessions, model configurations, and user settings.
 * Coordinates communication between the chat interface, settings panel, and backend API.
 *
 * @component
 * @returns {JSX.Element} The rendered application interface
 */
export default function App() {
    // Core state
    const [sessionId, setSessionId] = useState(null);
    const debouncedUpdateRef = useRef(null);
    const [error, setError] = useState(null);
    const [settingsVersion, setSettingsVersion] = useState(0);

    // UI state
    const [isOptionsOpen, setIsOptionsOpen] = useState(false);
    const [isStreaming, setIsStreaming] = useState(false);
    const [isLoading, setIsLoading] = useState(true);
    const [isInitialized, setIsInitialized] = useState(false);
    const [isReady, setIsReady] = useState(false);


    // Settings tracking
    const [persona, setPersona] = useState("");
    const [temperature, setTemperature] = useState(0.5);
    const [customPrompt, setCustomPrompt] = useState("");
    const [modelConfigs, setModelConfigs] = useState([]);
    const [modelName, setModelName] = useState("");
    const [modelParameters, setModelParameters] = useState({});
    const [selectedModel, setSelectedModel] = useState(null);

    // Data retrieval from backend
    const [personas, setPersonas] = useState([]);
    const [availableTools, setAvailableTools] = useState({
        essential_tools: [],
        groups: {},
        categories: []
    });
    const [activeTools, setActiveTools] = useState([]);

    // Fetch initial data (model configs, personas and tools)
    /**
     * Fetches and initializes all required data for the application startup.
     * Retrieves personas, available tools, and model configurations in parallel.
     * Sets up initial model, persona, and session configurations.
     *
     * @async
     * @function
     * @throws {Error} When any of the initial data fetches fail
     * @throws {Error} When no models are available
     */
    const fetchInitialData = async () => {
        try {
            setIsLoading(true);
            setIsInitialized(false);

            // Parallel data fetching
            const [personasResponse, toolsResponse, modelsResponse] = await Promise.all([
                fetch(`${API_URL}/personas`),
                fetch(`${API_URL}/get_tools`),
                fetch(`${API_URL}/models`)
            ]);

            if (!personasResponse.ok || !toolsResponse.ok || !modelsResponse.ok) {
                throw new Error('Failed to fetch initial data');
            }

            const [personasData, toolsData, modelsData] = await Promise.all([
                personasResponse.json(),
                toolsResponse.json(),
                modelsResponse.json()
            ]);

            // Set initial data
            setPersonas(personasData);
            setAvailableTools(toolsData);
            setModelConfigs(modelsData.models);

            // Initialize with first model if available
            if (modelsData.models.length > 0) {
                const initialModel = modelsData.models[0];
                setModelName(initialModel.id);
                setSelectedModel(initialModel);

                const initialParameters = {
                    temperature: initialModel.parameters?.temperature?.default ?? modelParameters.temperature,
                    reasoning_effort: initialModel.parameters?.reasoning_effort?.default ?? modelParameters.reasoning_effort
                };
                setModelParameters(initialParameters);

                // Set initial persona - prioritize 'default' or fallback to first
                if (personasData.length > 0) {
                    const defaultPersona = personasData.find(p => p.name === 'default');
                    const initialPersona = defaultPersona || personasData[0];
                    setPersona(initialPersona.name);
                    setCustomPrompt(initialPersona.content);
                }

                // Initialize session with initial model
                await initializeSession(false, initialModel);
                setIsInitialized(true);
            } else {
                throw new Error('No models available');
            }
        } catch
            (error) {
            console.error('Error fetching initial data:', error);
            setError(`Failed to load initial data: ${error.message}`);
        } finally {
            setIsLoading(false);
        }
    };

// Initialize (or reinitialize) session
    /**
     * Initializes or reinitializes a chat session with specified parameters.
     * Creates a new session with the backend and stores the session ID.
     *
     * @async
     * @function
     * @param {boolean} forceNew - Whether to force creation of a new session
     * @param {Object|null} initialModel - Model configuration to initialize with
     * @param {string} initialModel.id - Model identifier
     * @param {string} initialModel.backend - Backend service identifier
     * @throws {Error} When session initialization fails
     */
    const initializeSession = async (forceNew = false, initialModel = null) => {
        setIsReady(false);
        try {
            const currentModel = initialModel || modelConfigs.find(model => model.id === modelName);
            if (!currentModel) throw new Error('Invalid model configuration');

            const queryParams = new URLSearchParams({
                temperature: temperature.toString(),
                model_name: currentModel.id,
                backend: currentModel.backend,
                persona_name: persona || 'default'
            });

            const response = await fetch(`${API_URL}/initialize?${queryParams}`);
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

            const data = await response.json();

            if (data.session_id) {
                localStorage.setItem("session_id", data.session_id);
                setSessionId(data.session_id);
                setIsReady(true);
                setError(null);
            } else {
                throw new Error("No session_id in response");
            }
        } catch (error) {
            console.error("Session initialization failed:", error);
            setIsReady(false);
            setError(`Session initialization failed: ${error.message}`);
        }
    };

// Centralized fetchAgentTools function
    /**
     * Retrieves the current set of initialized tools for the active session.
     * Updates the activeTools state with the list of available tool class names.
     *
     * @async
     * @function
     * @requires sessionId
     * @requires isReady
     */
    const fetchAgentTools = async () => {
        if (!sessionId || !isReady) return;
        try {
            const response = await fetch(`${API_URL}/get_agent_tools/${sessionId}`);
            if (!response.ok) throw new Error(`Failed to fetch agent tools`);
            const data = await response.json();
            if (data.status === 'success' && Array.isArray(data.initialized_tools)) {
                setActiveTools(data.initialized_tools.map(tool => tool.class_name));
            }
        } catch (error) {
            console.error('Error fetching agent tools:', error);
        }
    };

//update agent settings
    /**
     * Updates agent settings based on the type of update requested.
     * Handles model changes, general settings updates, and parameter adjustments.
     * Implements debouncing for parameter updates to prevent excessive API calls.
     *
     * @async
     * @function
     * @param {'MODEL_CHANGE'|'SETTINGS_UPDATE'|'PARAMETER_UPDATE'} updateType - Type of settings update
     * @param {Object} values - New values to apply
     * @param {string} [values.modelName] - New model identifier for MODEL_CHANGE
     * @param {string} [values.persona_name] - New persona name for SETTINGS_UPDATE
     * @param {string} [values.customPrompt] - New custom prompt for SETTINGS_UPDATE
     * @param {number} [values.temperature] - New temperature for PARAMETER_UPDATE
     * @param {number} [values.reasoning_effort] - New reasoning effort for PARAMETER_UPDATE
     * @throws {Error} When settings update fails
     * @requires sessionId
     * @requires isReady
     */
    const updateAgentSettings = async (updateType, values) => {
        if (!sessionId || !isReady) return;

        try {
            switch (updateType) {
                case 'MODEL_CHANGE':
                    setIsReady(false);
                    const newModel = modelConfigs.find(model => model.id === values.modelName);
                    if (!newModel) throw new Error('Invalid model configuration');

                    // Initialize parameters for new model with defaults
                    const newParameters = {
                        temperature: newModel.parameters?.temperature?.default ?? modelParameters.temperature,
                        reasoning_effort: newModel.parameters?.reasoning_effort?.default ?? modelParameters.reasoning_effort
                    };

                    setModelName(values.modelName);
                    setSelectedModel(newModel);
                    setModelParameters(newParameters);

                    await initializeSession(false, newModel);
                    setSettingsVersion(v => v + 1);  // Trigger refresh
                    break;

                case 'SETTINGS_UPDATE':
                    const formData = new FormData();
                    formData.append('session_id', sessionId);

                    // Always include both persona and customPrompt in the update
                    const updatedPersona = values.persona_name || persona;
                    const updatedPrompt = values.customPrompt || customPrompt;

                    setPersona(updatedPersona);
                    setCustomPrompt(updatedPrompt);

                    formData.append('persona_name', updatedPersona);
                    formData.append('custom_prompt', updatedPrompt);

                    const response = await fetch(`${API_URL}/update_settings`, {
                        method: 'POST',
                        body: formData
                    });

                    if (!response.ok) {
                        throw new Error('Failed to update settings');
                    }

                    setSettingsVersion(v => v + 1);  // Trigger refresh
                    break;

                case 'PARAMETER_UPDATE':
                    if (debouncedUpdateRef.current) {
                        clearTimeout(debouncedUpdateRef.current);
                    }

                    debouncedUpdateRef.current = setTimeout(async () => {
                        const formData = new FormData();
                        formData.append('session_id', sessionId);

                        // Create updated parameters object
                        const updatedParameters = {...modelParameters};

                        if ('temperature' in values) {
                            updatedParameters.temperature = values.temperature;
                            formData.append('temperature', values.temperature);
                        }
                        if ('reasoning_effort' in values) {
                            updatedParameters.reasoning_effort = values.reasoning_effort;
                            formData.append('reasoning_effort', values.reasoning_effort);
                        }

                        // Update local state
                        setModelParameters(updatedParameters);

                        // Send to backend
                        await fetch(`${API_URL}/update_settings`, {
                            method: 'POST',
                            body: formData
                        });
                        setSettingsVersion(v => v + 1);  // Trigger refresh
                    }, 300);
                    break;
            }
        } catch (error) {
            setError(`Failed to update settings: ${error.message}`);
        }
    };

// Handle equipping tools
    /**
     * Updates the set of equipped tools for the current session.
     * Sends tool configuration to the backend and refreshes the active tools list.
     *
     * @async
     * @function
     * @param {string[]} tools - Array of tool identifiers to equip
     * @throws {Error} When tool equipment update fails
     * @requires sessionId
     */
    const handleEquipTools = async (tools) => {
        const formData = new FormData();
        formData.append('session_id', sessionId);
        formData.append('tools', JSON.stringify(tools));
        try {
            const response = await fetch(`${API_URL}/update_tools`, {
                method: 'POST',
                body: formData
            });
            if (!response.ok) throw new Error("Failed to equip tools");
            await fetchAgentTools();
        } catch (error) {
            console.error("Failed to equip tools:", error);
            throw error;
        }
    };

    /**
     * Updates the streaming status of the chat interface.
     * Controls loading indicators and UI state during message processing.
     *
     * @function
     * @param {boolean} status - Current processing status
     */
    const handleProcessingStatus = (status) => {
        // console.log('Processing status changed:', status); // Debug log
        setIsStreaming(status);
    };

// Handle session deletion
    /**
     * Handles cleanup when sessions are deleted.
     * Removes local storage data and resets relevant state variables.
     *
     * @function
     */
    const handleSessionsDeleted = () => {
        localStorage.removeItem("session_id");
        setSessionId(null);
        setIsReady(false);
        setActiveTools([]);
        setError(null);
    };

// Fetch initial data on mount
    useEffect(() => {
        fetchInitialData();
    }, []);

// Update agent tools when session, readiness, or model changes
    useEffect(() => {
        if (sessionId && isReady) {
            fetchAgentTools();
        }
    }, [sessionId, isReady, modelName]);

    return (
        <div className="min-h-screen bg-gradient-to-b from-gray-50 to-gray-100">
            <div className="container mx-auto px-6 py-4 max-w-7xl h-screen flex flex-col">
                <div className="flex flex-col space-y-4 mb-4">
                    <div className="flex items-center justify-between">
                        <h1 className="text-2xl font-bold text-gray-900">
                            Agent C Conversational Interface
                        </h1>
                    </div>

                    {error && (
                        <Alert variant="destructive" className="mb-4">
                            <AlertDescription className="flex justify-between items-center">
                                {error}
                                <button
                                    onClick={() => setError(null)}
                                    className="ml-2 text-sm underline hover:no-underline"
                                >
                                    Dismiss
                                </button>
                            </AlertDescription>
                        </Alert>
                    )}

                    {isReady && (
                        <StatusBar
                            isReady={isReady}
                            activeTools={activeTools}
                            onSessionsDeleted={handleSessionsDeleted}
                            sessionId={sessionId}
                            settingsVersion={settingsVersion}
                            isProcessing={isStreaming}
                        />
                    )}
                </div>

                {!isLoading && sessionId && isInitialized ? (
                    <div className="flex-1 flex flex-col min-h-0">
                        <CollapsibleOptions
                            isOpen={isOptionsOpen}
                            setIsOpen={setIsOptionsOpen}
                            persona={persona}
                            personas={personas}
                            availableTools={availableTools}
                            customPrompt={customPrompt}
                            temperature={temperature}
                            modelName={modelName}
                            modelConfigs={modelConfigs}
                            sessionId={sessionId}
                            isReady={isReady}
                            onEquipTools={handleEquipTools}
                            activeTools={activeTools}
                            modelParameters={modelParameters}
                            selectedModel={selectedModel}
                            onUpdateSettings={updateAgentSettings}
                            isInitialized={isInitialized}
                            onProcessingStatus={(status) => setIsStreaming(status)}
                        />

                        <div className="flex-1 min-h-0">
                            <ChatInterface
                                sessionId={sessionId}
                                customPrompt={customPrompt}
                                modelName={modelName}
                                modelParameters={modelParameters}
                                selectedModel={selectedModel}
                                onProcessingStatus={handleProcessingStatus}
                            />
                        </div>
                    </div>
                ) : (
                    <div className="flex-1 flex items-center justify-center">
                        <div className="text-lg text-muted-foreground animate-pulse">
                            Initializing session...
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
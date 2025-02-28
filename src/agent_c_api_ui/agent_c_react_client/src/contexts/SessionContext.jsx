import React, {createContext, useState, useEffect, useRef} from 'react';
import {API_URL} from '@/config/config';

if (!API_URL) {
    console.error('API_URL is not defined! Environment variables may not be loading correctly.');
    console.log('Current environment variables:', {
        'import.meta.env.VITE_API_URL': import.meta.env.VITE_API_URL,
        'process.env.VITE_API_URL': process.env?.VITE_API_URL,
        'NODE_ENV': process.env?.NODE_ENV
    });
}

export const SessionContext = createContext();

export const SessionProvider = ({children}) => {
    // Global session & UI state
    const [sessionId, setSessionId] = useState(null);
    const debouncedUpdateRef = useRef(null);
    const [error, setError] = useState(null);
    const [settingsVersion, setSettingsVersion] = useState(0);
    const [isOptionsOpen, setIsOptionsOpen] = useState(false);
    const [isStreaming, setIsStreaming] = useState(false);
    const [isLoading, setIsLoading] = useState(true);
    const [isInitialized, setIsInitialized] = useState(false);
    const [isReady, setIsReady] = useState(false);

    // Agent settings & configuration
    const [persona, setPersona] = useState("");
    const [temperature, setTemperature] = useState(null);
    const [customPrompt, setCustomPrompt] = useState("");
    const [modelConfigs, setModelConfigs] = useState([]);
    const [modelName, setModelName] = useState("");
    const [modelParameters, setModelParameters] = useState({});
    const [selectedModel, setSelectedModel] = useState(null);

    // Data from backend
    const [personas, setPersonas] = useState([]);
    const [availableTools, setAvailableTools] = useState({
        essential_tools: [],
        groups: {},
        categories: []
    });
    const [activeTools, setActiveTools] = useState([]);

    // --- Business Logic Functions ---
    const checkResponse = async (response, endpoint) => {
        const contentType = response.headers.get("content-type");
        console.log(`Response from ${endpoint}:`, {
            status: response.status,
            contentType,
            ok: response.ok
        });

        if (!response.ok) {
            // Try to get error details
            let errorText;
            try {
                errorText = await response.text();
            } catch (e) {
                errorText = 'Could not read error response';
            }
            throw new Error(`${endpoint} failed: ${response.status} - ${errorText}`);
        }

        if (!contentType?.includes('application/json')) {
            throw new Error(`${endpoint} returned non-JSON content-type: ${contentType}`);
        }

        const text = await response.text();
        console.log(`Raw response from ${endpoint}:`, text);

        try {
            return JSON.parse(text);
        } catch (e) {
            console.error(`Failed to parse JSON from ${endpoint}:`, text);
            throw new Error(`Invalid JSON from ${endpoint}: ${e.message}`);
        }
    };

    // Fetch initial data (personas, tools, models)
    const fetchInitialData = async () => {
        try {
            console.log('Starting fetchInitialData with API_URL:', API_URL);
            if (!API_URL) {
                throw new Error('API_URL is undefined. Please check your environment variables.');
            }
            setIsLoading(true);
            setIsInitialized(false);

            // Parallel fetching
            const [personasResponse, toolsResponse, modelsResponse] = await Promise.all([
                fetch(`${API_URL}/personas`),
                fetch(`${API_URL}/tools`),
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
            // console.log('Fetched initial data:', {personasData, toolsData, modelsData});
            setPersonas(personasData);
            setAvailableTools(toolsData);
            setModelConfigs(modelsData.models);


            if (modelsData.models.length > 0) {
                // console.log('Initializing session with model:', modelsData.models[0]);
                const initialModel = modelsData.models[0];
                setModelName(initialModel.id);
                setSelectedModel(initialModel);

                // non-reasoning model parameters
                const initialTemperature = initialModel.parameters?.temperature?.default ?? 0.3;
                // openai reasoning model parameters
                const initialReasoningEffort = initialModel.parameters?.reasoning_effort?.default;

                // claude reasoning model parameters
                const hasExtendedThinking = !!initialModel.parameters?.extended_thinking;
                const initialExtendedThinking = hasExtendedThinking ?
                    initialModel.parameters.extended_thinking.enabled === true : false;
                const initialBudgetTokens = hasExtendedThinking && initialExtendedThinking ?
                    initialModel.parameters.extended_thinking.budget_tokens?.default || 4000 : 0;

                const initialParameters = {
                    temperature: initialTemperature,
                    reasoning_effort: initialReasoningEffort,
                    extended_thinking: initialExtendedThinking,
                    budget_tokens: initialBudgetTokens
                };

                setModelParameters(initialParameters);

                // Choose a default persona (or fall back to the first)
                if (personasData.length > 0) {
                    // console.log('Setting initial persona:', personasData[0]);
                    const defaultPersona = personasData.find(p => p.name === 'default');
                    const initialPersona = defaultPersona || personasData[0];
                    setPersona(initialPersona.name);
                    setCustomPrompt(initialPersona.content);
                }

                // Initialize a session with the initial model
                // console.log('Initializing session with initial model:', initialModel);
                await initializeSession(false, initialModel);
                setIsInitialized(true);
            } else {
                throw new Error('No models available');
            }
        } catch (err) {
            console.error('Error fetching initial data:', err);
            setError(`Failed to load initial data: ${err.message}`);
        } finally {
            setIsLoading(false);
        }
    };

    // Initialize (or reinitialize) a session
    const initializeSession = async (forceNew = false, initialModel = null) => {
        setIsReady(false);
        try {
            // When initialModel is passed as an object with custom fields, extract the real model
            let currentModel;
            if (initialModel) {
                // If initialModel has customPrompt/persona_name properties but not a proper model structure
                if ('customPrompt' in initialModel || 'persona_name' in initialModel) {
                    // Use the id from the passed model object
                    currentModel = modelConfigs.find(model => model.id === initialModel.id);
                    console.log('Using model from configs:', currentModel?.id);
                } else {
                    currentModel = initialModel;
                }
            } else {
                currentModel = modelConfigs.find(model => model.id === modelName);
            }

            if (!currentModel) {
                console.error('Invalid model configuration', {modelName, initialModel});
                throw new Error('Invalid model configuration');
            }

            // Start with the required parameters
            const queryParams = new URLSearchParams({
                model_name: currentModel.id,
                backend: currentModel.backend,
                persona_name: persona || 'default'
            });

            // Add custom prompt if available
            if (customPrompt) {
                queryParams.append('custom_prompt', customPrompt);
            }


            // Conditionally add temperature if model supports it
            if (currentModel.parameters?.temperature) {
                const currentTemp = temperature ?? currentModel.parameters.temperature.default;
                queryParams.append('temperature', currentTemp.toString());
                console.log(`Initializing with temperature=${currentTemp}`);
            }

            // Handle Claude extended thinking parameters
            const hasExtendedThinking = !!currentModel.parameters?.extended_thinking;
            if (hasExtendedThinking) {
                // Determine if extended thinking should be enabled
                const extendedThinkingDefault = Boolean(currentModel.parameters.extended_thinking.enabled) === true;
                console.log(`Default extended_thinking from config: ${extendedThinkingDefault}`);

                const extendedThinking = initialModel ?
                    extendedThinkingDefault :
                    (modelParameters.extended_thinking !== undefined ?
                        modelParameters.extended_thinking : extendedThinkingDefault);

                // Add extended_thinking parameter
                queryParams.append('extended_thinking', extendedThinking.toString());

                // Set budget tokens based on whether extended thinking is enabled
                // Get budget tokens default
                const defaultBudgetTokens = parseInt(
                    currentModel.parameters.extended_thinking.budget_tokens?.default || 5000
                );
                console.log(`Default budget_tokens from config: ${defaultBudgetTokens}`);

                const budgetTokens = initialModel ?
                    (extendedThinking ? defaultBudgetTokens : 0) :
                    (extendedThinking ?
                        (modelParameters.budget_tokens !== undefined ?
                            modelParameters.budget_tokens : defaultBudgetTokens) : 0);
                1


                // Add budget_tokens parameter
                queryParams.append('budget_tokens', budgetTokens.toString());

                // Update local state for UI consistency
                setModelParameters(prev => ({
                    ...prev,
                    extended_thinking: extendedThinking,
                    budget_tokens: budgetTokens
                }));

                console.log(`Initializing with extended_thinking=${extendedThinking}, budget_tokens=${budgetTokens}`);
            }

            // Handle OpenAI reasoning effort parameter if supported
            if (currentModel.parameters?.reasoning_effort) {
                const reasoningEffortDefault = currentModel.parameters.reasoning_effort.default;
                const reasoningEffort = modelParameters.reasoning_effort !== undefined ?
                    modelParameters.reasoning_effort : reasoningEffortDefault;

                queryParams.append('reasoning_effort', reasoningEffort);

                // Update local state
                setModelParameters(prev => ({
                    ...prev,
                    reasoning_effort: reasoningEffort
                }));

                console.log(`Initializing with reasoning_effort=${reasoningEffort}`);
            }

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
        } catch (err) {
            console.error("Session initialization failed:", err);
            setIsReady(false);
            setError(`Session initialization failed: ${err.message}`);
        }
    };

    // Fetch agent tools for the current session
    const fetchAgentTools = async () => {
        if (!sessionId || !isReady) return;
        try {
            const response = await fetch(`${API_URL}/get_agent_tools/${sessionId}`);
            if (!response.ok) throw new Error('Failed to fetch agent tools');
            const data = await response.json();
            if (data.status === 'success' && Array.isArray(data.initialized_tools)) {
                setActiveTools(data.initialized_tools.map(tool => tool.class_name));
            }
        } catch (err) {
            console.error('Error fetching agent tools:', err);
        }
    };

    // Update agent settings (model change, settings update, parameter update)
    const updateAgentSettings = async (updateType, values) => {
        if (!sessionId || !isReady) return;
        try {
            switch (updateType) {
                case 'MODEL_CHANGE': {
                    setIsReady(false);
                    const newModel = modelConfigs.find(model => model.id === values.modelName);
                    if (!newModel) throw new Error('Invalid model configuration');

                    // Update the state first
                    setModelName(values.modelName);
                    setSelectedModel(newModel);

                    // Create the persona/model object
                    const modelWithPersona = {
                        ...newModel,
                        persona_name: persona,
                        custom_prompt: customPrompt
                    };

                    // Determine extended thinking parameters based on new model
                    const hasExtendedThinking = !!newModel.parameters?.extended_thinking;
                    const extendedThinkingDefault = hasExtendedThinking ?
                        newModel.parameters.extended_thinking.enabled === true : false;

                    const budgetTokensDefault = hasExtendedThinking && extendedThinkingDefault ?
                        newModel.parameters.extended_thinking.budget_tokens?.default || 5000 : 0;

                    const newParameters = {
                        temperature: newModel.parameters?.temperature?.default ?? modelParameters.temperature,
                        reasoning_effort: newModel.parameters?.reasoning_effort?.default ?? modelParameters.reasoning_effort,
                        extended_thinking: extendedThinkingDefault,
                        budget_tokens: budgetTokensDefault
                    };

                    setModelParameters(newParameters);
                    setSelectedModel(newModel);


                    await initializeSession(false, modelWithPersona);
                    setSettingsVersion(v => v + 1);
                    break;
                }
                case 'SETTINGS_UPDATE': {
                    const formData = new FormData();
                    formData.append('session_id', sessionId);
                    formData.append('model_name', modelName);
                    formData.append('backend', selectedModel?.backend);
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
                    if (!response.ok) throw new Error('Failed to update settings');
                    setSettingsVersion(v => v + 1);
                    break;
                }
                case 'PARAMETER_UPDATE': {
                    if (debouncedUpdateRef.current) {
                        clearTimeout(debouncedUpdateRef.current);
                    }
                    debouncedUpdateRef.current = setTimeout(async () => {
                        const formData = new FormData();
                        formData.append('session_id', sessionId);
                        const updatedParameters = {...modelParameters};

                        // non-reasoning model parameters
                        if ('temperature' in values) {
                            updatedParameters.temperature = values.temperature;
                            formData.append('temperature', values.temperature);
                        }

                        // openai reasoning model parameters
                        if ('reasoning_effort' in values) {
                            updatedParameters.reasoning_effort = values.reasoning_effort;
                            formData.append('reasoning_effort', values.reasoning_effort);
                        }
                        // claude reasoning model parameters
                        if ('extended_thinking' in values) {
                            updatedParameters.extended_thinking = values.extended_thinking;
                            formData.append('extended_thinking', values.extended_thinking);

                            // If extended thinking is disabled, set budget tokens to 0
                            if (!values.extended_thinking) {
                                updatedParameters.budget_tokens = 0;
                                formData.append('budget_tokens', '0');
                            }
                        }

                        if ('budget_tokens' in values) {
                            // Only update budget tokens if extended thinking is enabled
                            if (updatedParameters.extended_thinking) {
                                updatedParameters.budget_tokens = values.budget_tokens;
                                formData.append('budget_tokens', values.budget_tokens);
                            }
                        }

                        setModelParameters(updatedParameters);
                        await fetch(`${API_URL}/update_settings`, {
                            method: 'POST',
                            body: formData
                        });
                        setSettingsVersion(v => v + 1);
                    }, 300);
                    break;
                }
                default:
                    break;
            }
        } catch (err) {
            setError(`Failed to update settings: ${err.message}`);
        }
    };

    // Handle equipping tools
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
            setSettingsVersion(v => v + 1);
        } catch (err) {
            console.error("Failed to equip tools:", err);
            throw err;
        }
    };

    // Update processing status (for loading/spinner UI)
    const handleProcessingStatus = (status) => {
        setIsStreaming(status);
    };

    // Handle session deletion
    const handleSessionsDeleted = () => {
        localStorage.removeItem("session_id");
        setSessionId(null);
        setIsReady(false);
        setActiveTools([]);
        setError(null);
    };

    // --- Effects ---
    useEffect(() => {
        fetchInitialData();
    }, []);

    useEffect(() => {
        if (sessionId && isReady) {
            fetchAgentTools();
        }
    }, [sessionId, isReady, modelName]);

    return (
        <SessionContext.Provider
            value={{
                sessionId,
                error,
                settingsVersion,
                isOptionsOpen,
                setIsOptionsOpen,
                isStreaming,
                isLoading,
                isInitialized,
                isReady,
                persona,
                temperature,
                customPrompt,
                modelConfigs,
                modelName,
                modelParameters,
                selectedModel,
                personas,
                availableTools,
                activeTools,
                fetchAgentTools,
                updateAgentSettings,
                handleEquipTools,
                handleProcessingStatus,
                handleSessionsDeleted
            }}
        >
            {children}
        </SessionContext.Provider>
    );
};
import React, {useState, useEffect} from 'react';
import {Settings} from 'lucide-react';
import {Tooltip, TooltipContent, TooltipProvider, TooltipTrigger} from "@/components/ui/tooltip";
import * as Portal from '@radix-ui/react-portal';

const API_URL = import.meta.env.VITE_API_URL;

/**
 * AgentConfigDisplay is a component that fetches and displays agent configuration
 * information in a tooltip interface.
 *
 * @component
 * @param {Object} props
 * @param {string} props.sessionId - Session identifier for fetching configuration
 * @param {string} [props.className=""] - Additional CSS classes to apply
 * @param {number} props.settingsVersion - Version identifier that triggers config refresh
 */
const AgentConfigDisplay = ({sessionId, className = "", settingsVersion}) => {
    const [config, setConfig] = useState(null);
    const [error, setError] = useState(null);

    // Debug log when props change
    // useEffect(() => {
    //   console.log('AgentConfigDisplay props updated:', {
    //     sessionId,
    //     settingsVersion,
    //     currentConfig: config
    //   });
    // }, [sessionId, settingsVersion, config]);

    useEffect(() => {
        /**
         * Fetches the agent configuration from the API
         * @async
         * @function
         * @throws {Error} When the API request fails
         */
        const fetchConfig = async () => {
            try {
                if (!sessionId) {
                    console.log('No sessionId provided, skipping fetch');
                    return;
                }

                console.log('Fetching config... settingsVersion:', settingsVersion);
                const url = `${API_URL}/get_agent_config/${sessionId}`;

                const response = await fetch(url, {
                    // Add cache-busting query parameter
                    headers: {
                        'Cache-Control': 'no-cache',
                        'Pragma': 'no-cache'
                    }
                });

                if (!response.ok) {
                    throw new Error(`Failed to fetch agent configuration: ${response.status}`);
                }

                const data = await response.json();
                console.log('Received new config for agent:', data);

                // Compare with previous config to see what changed
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

                setConfig(data.config);
                setError(null);
            } catch (err) {
                setError(err.message);
                console.error('Error fetching agent config:', err);
            }
        };

        /**
         * Handles API error states
         * @param {Error} err - The error object from the API call
         */
        const handleError = (err) => {
            setError(err.message);
            console.error('Error fetching agent config:', err);
        };

        console.log('Triggering config fetch, settingsVersion:', settingsVersion);
        fetchConfig();
    }, [sessionId, settingsVersion]);

    if (!sessionId) {
        return null;
    }

    if (error) {
        console.error("Error in AgentConfigDisplay:", error);
        return null;
    }

    if (!config) {
        return (
            <div className={`inline-flex items-center cursor-help ${className}`}>
                <Settings className="w-4 h-4 text-gray-400"/>
                <span className="ml-1 text-sm text-gray-400">Loading...</span>
            </div>
        );
    }

    // Format the configuration information for display
    /**
     * Formats the configuration data for display in the tooltip
     * @type {Record<string, string|number>}
     */
    const configDisplay = {
        "Model": config.model_info?.name,
        "Backend": config.backend,
        ...(config.model_info?.temperature !== undefined && {
            "Temperature": config.model_info?.temperature.toFixed(2)
        }),
        ...(config.model_info?.reasoning_effort && {
            "Reasoning Effort": config.model_info?.reasoning_effort
        }),
        "Persona": config.persona_name,
        "Active Tools": config.initialized_tools.length,
        "Session ID": config.session_id
    };

    return (
        <TooltipProvider>
            <Tooltip>
                <TooltipTrigger asChild>
                    <div className={`inline-flex items-center cursor-help ${className}`}>
                        <Settings className="w-4 h-4 text-blue-500 hover:text-blue-600"/>
                        <span className="ml-1 text-sm text-blue-500 hover:text-blue-600">
              Current Config
            </span>
                    </div>
                </TooltipTrigger>
                <Portal.Root>
                    <TooltipContent side="right" className="w-80 bg-white border shadow-md relative z-50">
                        <div className="p-2">
                            <h4 className="font-semibold mb-2 text-gray-900">Agent Configuration</h4>
                            <div className="space-y-1 bg-white">
                                {Object.entries(configDisplay).map(([key, value]) => (
                                    <div key={key} className="grid grid-cols-2 text-sm">
                                        <span className="text-gray-500">{key}:</span>
                                        <span className="font-medium text-gray-900">{value}</span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </TooltipContent>
                </Portal.Root>
            </Tooltip>
        </TooltipProvider>
    );
};

export default AgentConfigDisplay;
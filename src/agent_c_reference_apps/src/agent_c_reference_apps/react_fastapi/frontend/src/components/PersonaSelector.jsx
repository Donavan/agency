import React, {useCallback, useEffect, useState} from 'react';
import {Card, CardContent, CardHeader, CardTitle} from '@/components/ui/card';
import {Textarea} from '@/components/ui/textarea';
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from '@/components/ui/select';
import {Label} from '@/components/ui/label';
import {
    Tooltip,
    TooltipContent,
    TooltipProvider,
    TooltipTrigger,
} from "@/components/ui/tooltip";
import ModelParameterControls from './ModelParameterControls';


/**
 * PersonaSelector is a component that manages AI model settings, persona selection,
 * and custom prompts through a user interface.
 *
 * @component
 * @param {Object} props
 * @param {string} props.persona - Currently selected persona identifier
 * @param {Array<{name: string, content: string}>} props.personas - Available personas
 * @param {string} props.customPrompt - Custom instructions for the AI model
 * @param {string} props.modelName - Current model identifier
 * @param {Array<ModelConfig>} props.modelConfigs - Available model configurations
 * @param {string} props.sessionId - Current session identifier
 * @param {Object} props.modelParameters - Current model parameters
 * @param {ModelConfig} props.selectedModel - Currently selected model configuration
 * @param {Function} props.onUpdateSettings - Callback for settings updates
 * @param {boolean} props.isInitialized - Component initialization status
 */
function PersonaSelector({
                             persona,
                             personas,
                             customPrompt,
                             modelName,
                             modelConfigs,
                             sessionId,
                             modelParameters,
                             selectedModel,
                             onUpdateSettings,
                             isInitialized
                         }) {
    // Local UI state only
    const [error, setError] = useState(null);
    const [selectedPersona, setSelectedPersona] = useState(persona || 'default');
    const [localCustomPrompt, setLocalCustomPrompt] = useState(customPrompt);

    // Sync local UI state with prop
    useEffect(() => {
        if (isInitialized && persona) {
            setSelectedPersona(persona);
            setLocalCustomPrompt(customPrompt);
        }
    }, [isInitialized, persona, customPrompt]);

    /**
     * Handles persona selection changes
     * @param {string} value - Selected persona identifier
     */
    const handlePersonaChange = useCallback((value) => {
        const selectedPersonaData = personas.find(p => p.name === value);
        if (selectedPersonaData) {
            setSelectedPersona(value); // Update local state immediately
            // Delay the settings update slightly to ensure state consistency
            setTimeout(() => {
                onUpdateSettings('SETTINGS_UPDATE', {
                    persona_name: value,
                    customPrompt: selectedPersonaData.content
                });
            }, 0);
        }
    }, [personas, onUpdateSettings]);

    /**
     * Handles changes to the custom prompt textarea
     * @param {React.ChangeEvent<HTMLTextAreaElement>} e - Change event
     */
    const handleCustomPromptChange = useCallback((e) => {
        // Update local state immediately for responsive UI
        // console.log('Custom Prompt Change:', e.target.value);
        setLocalCustomPrompt(e.target.value);
    }, []);

    /**
     * Triggers settings update when custom prompt editing is complete
     */
    const handleCustomPromptBlur = useCallback(() => {
        // Only send update if the value has actually changed
        if (localCustomPrompt !== customPrompt) {
            console.log('User Changed Custom Prompt:', localCustomPrompt);
            onUpdateSettings('SETTINGS_UPDATE', {
                customPrompt: localCustomPrompt
            });
        }
    }, [localCustomPrompt, customPrompt, onUpdateSettings]);

    /**
     * Handles model parameter changes
     * @param {string} paramName - Name of the parameter to update
     * @param {any} value - New parameter value
     */
    const handleParameterChange = useCallback((paramName, value) => {
        onUpdateSettings('PARAMETER_UPDATE', {
            [paramName]: value
        });
    }, [onUpdateSettings]);

    /**
     * Handles model selection changes
     * @param {string} selectedValue - Selected model identifier
     */
    const handleModelChange = useCallback((selectedValue) => {
        const model = modelConfigs.find(model => model.id === selectedValue);
        if (model) {
            onUpdateSettings('MODEL_CHANGE', {
                modelName: model.id,
                backend: model.backend
            });
        }
    }, [modelConfigs, onUpdateSettings]);

    return (
        <Card className="w-full">
            <CardHeader>
                <CardTitle>Settings</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
                {/* Persona Selection */}
                <div className="space-y-2">
                    <Label htmlFor="persona-select">Load Persona Prompt</Label>
                    <Select value={selectedPersona}
                            onValueChange={handlePersonaChange}>
                        <SelectTrigger
                            id="persona-select"
                            className="rounded-xl border-gray-200 bg-white/50 backdrop-blur-sm transition-colors hover:bg-white/80 focus:border-blue-300 focus:ring focus:ring-blue-200 focus:ring-opacity-50"
                        >
                            <SelectValue placeholder="Select a persona"/>
                        </SelectTrigger>
                        <SelectContent className="bg-white/95 backdrop-blur-sm border shadow-lg rounded-xl">
                            {personas.map((p) => (
                                <SelectItem
                                    key={p.name}
                                    value={p.name}
                                    className="hover:bg-blue-50/80 focus:bg-blue-50 transition-colors rounded-lg mx-1 my-0.5"
                                >
                                    {p.name}
                                </SelectItem>
                            ))}
                        </SelectContent>
                    </Select>
                    {error && (
                        <div className="text-sm text-red-500 mt-1">
                            Error: {error}
                        </div>
                    )}
                </div>

                {/* Custom Instructions */}
                <div className="space-y-2">
                    <Label htmlFor="custom-prompt">Customize Persona Instructions</Label>
                    <Textarea
                        id="custom-prompt"
                        value={localCustomPrompt}
                        onChange={handleCustomPromptChange}
                        onBlur={handleCustomPromptBlur}
                        className="min-h-[100px] resize-y rounded-xl border-gray-200 bg-white/50 backdrop-blur-sm transition-colors hover:bg-white/80 focus:border-blue-300 focus:ring focus:ring-blue-200 focus:ring-opacity-50"
                        placeholder="You are a helpful assistant."
                    />
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    {/* Model Selection */}
                    <div className="space-y-2">
                        <Label htmlFor="model-select">Model</Label>
                        <Select
                            value={modelName}
                            onValueChange={handleModelChange}
                        >
                            <SelectTrigger
                                id="model-select"
                                className="rounded-xl border-gray-200 bg-white/50 backdrop-blur-sm transition-colors hover:bg-white/80 focus:border-blue-300 focus:ring focus:ring-blue-200 focus:ring-opacity-50"
                            >
                                <SelectValue placeholder="Select model"/>
                            </SelectTrigger>
                            <SelectContent className="bg-white/95 backdrop-blur-sm border shadow-lg rounded-xl">
                                {Object.entries(modelConfigs.reduce((acc, model) => {
                                    if (!acc[model.backend]) acc[model.backend] = [];
                                    acc[model.backend].push(model);
                                    return acc;
                                }, {})).map(([vendor, vendorModels]) => (
                                    <React.Fragment key={vendor}>
                                        <SelectItem
                                            value={`header-${vendor}`}
                                            disabled
                                            className="opacity-50 pointer-events-none px-2 py-1.5 text-sm font-semibold text-muted-foreground"
                                        >
                                            {vendor.toUpperCase()}
                                        </SelectItem>
                                        {vendorModels.map((model) => (
                                            <TooltipProvider key={model.id}>
                                                <Tooltip>
                                                    <TooltipTrigger asChild>
                                                        <SelectItem
                                                            value={model.id}
                                                            className="hover:bg-blue-50/80 focus:bg-blue-50 transition-colors rounded-lg mx-1 my-0.5 pl-4"
                                                        >
                                                            {model.label}
                                                        </SelectItem>
                                                    </TooltipTrigger>
                                                    <TooltipContent>
                                                        <p>{model.description}</p>
                                                        <p className="text-xs text-muted-foreground mt-1">
                                                            Type: {model.model_type}
                                                        </p>
                                                    </TooltipContent>
                                                </Tooltip>
                                            </TooltipProvider>
                                        ))}
                                    </React.Fragment>
                                ))}
                            </SelectContent>
                        </Select>
                    </div>

                    {/* Model Parameters */}
                    <ModelParameterControls
                        selectedModel={selectedModel}
                        onParameterChange={handleParameterChange}
                        currentParameters={modelParameters}
                    />
                </div>
            </CardContent>
        </Card>
    );
}

export default PersonaSelector;
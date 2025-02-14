import React, {useState, useEffect} from 'react';
import {Label} from '@/components/ui/label';
import {Slider} from '@/components/ui/slider';
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from '@/components/ui/select';

/**
 * ModelParameterControls provides an interface for adjusting AI model parameters
 * including temperature and reasoning effort controls.
 *
 * @component
 * @param {Object} props
 * @param {Object} props.selectedModel - Selected model configuration
 * @param {Object} props.selectedModel.parameters - Model parameter configurations
 * @param {Function} props.onParameterChange - Callback for parameter changes
 * @param {Object} [props.currentParameters={}] - Current parameter values
 */
const ModelParameterControls = ({
                                    selectedModel,
                                    onParameterChange,
                                    currentParameters = {},
                                }) => {
    const [temperature, setTemperature] = useState(currentParameters?.temperature ?? 0.5);
    const [reasoningEffort, setReasoningEffort] = useState(currentParameters?.reasoning_effort ?? 'medium');
    const [localTemperature, setLocalTemperature] = useState(currentParameters?.temperature ?? 0.5);

    /**
     * Handles real-time temperature slider changes
     * @param {number[]} value - Array containing single temperature value
     */
    const handleTemperatureChange = (value) => {
        // console.log('Temperature Change:', value);
        const temp = value[0];
        setLocalTemperature(temp);
    };

    /**
     * Commits temperature changes to the backend
     * @param {number[]} value - Array containing single temperature value
     */
    const handleTemperatureCommit = (value) => {
        // console.log('Temperature Commit:', value);
        const temp = value[0];
        setTemperature(temp);  // Update the main temperature state
        onParameterChange('temperature', temp);
    };

    useEffect(() => {
        // console.log('Current Parameters:', currentParameters);
        // console.log('Selected Model Parameters:', selectedModel?.parameters);
        if (selectedModel?.parameters) {
            const defaultTemp = selectedModel.parameters?.temperature?.default ?? 0.5;
            const defaultEffort = selectedModel.parameters?.reasoning_effort?.default ?? 'low';
            console.log('Default Temperature:', defaultTemp);
            setLocalTemperature(currentParameters?.temperature ?? defaultTemp);
            // setTemperature(currentParameters?.temperature ?? defaultTemp);
            setReasoningEffort(currentParameters?.reasoning_effort ?? defaultEffort);
        }
    }, [selectedModel, currentParameters]);

    if (!selectedModel?.parameters) return null;

    /**
     * Temperature slider configuration
     * @type {Object}
     * @property {number} min - Minimum temperature value
     * @property {number} max - Maximum temperature value
     * @property {number} step - Temperature adjustment increment, fixed at 0.1
     * @property {number} default - Default temperature value
     */
    const temperatureConfig = {
        min: selectedModel.parameters?.temperature?.min,
        max: selectedModel.parameters?.temperature?.max,
        step: 0.1,
        default: selectedModel.parameters?.temperature?.default
    };
    // console.log('Temperature Config:', temperatureConfig);
    // console.log('Slider Props:', {
    //     min: temperatureConfig.min,
    //     max: temperatureConfig.max,
    //     step: temperatureConfig.step,
    //     value: localTemperature
    // });

    /**
     * Retrieves reasoning effort options from model configuration
     * @returns {string[]|null} Array of reasoning effort options or null if not configured
     */
    const getReasoningEffortOptions = () => {
        const config = selectedModel.parameters?.reasoning_effort;
        if (!config) return null;
        if (Array.isArray(config.options)) return config.options;
        if (Array.isArray(config)) return config;
        if (typeof config === 'object') {
            return config.values || config.choices || ['low', 'medium', 'high'];
        }
        return ['low', 'medium', 'high'];
    };

    const reasoningEffortOptions = getReasoningEffortOptions();

    /**
     * Handles changes to reasoning effort selection
     * @param {string} value - Selected reasoning effort level
     */
    const handleReasoningEffortChange = (value) => {
        setReasoningEffort(value);
        onParameterChange('reasoning_effort', value);
    };

    return (
        <div className="space-y-4">
            {selectedModel.parameters?.temperature && (
                <div className="space-y-2">
                    <div className="flex items-center justify-between">
                        <Label className="text-sm font-medium">Temperature</Label>
                        <span className="px-2 py-1 bg-blue-100 text-blue-700 rounded-xl text-sm font-medium">
                            {localTemperature.toFixed(1)}
                        </span>
                    </div>
                    <div className="relative pt-1">
                        <div className="flex justify-between text-xs text-muted-foreground mb-2">
                            <span className="text-left">Focused</span>
                            <span className="text-center">Balanced</span>
                            <span className="text-right">Creative</span>
                        </div>
                        <div className="absolute w-full flex justify-between px-[2px] pointer-events-none">
                            <div className="w-[2px] h-2 bg-muted-foreground/20"></div>
                            <div className="w-[2px] h-2 bg-muted-foreground/20"></div>
                            <div className="w-[2px] h-2 bg-muted-foreground/20"></div>
                            <div className="w-[2px] h-2 bg-muted-foreground/20"></div>
                            <div className="w-[2px] h-2 bg-muted-foreground/20"></div>
                        </div>
                        <Slider
                            id="temperature-slider"
                            min={temperatureConfig.min}
                            max={temperatureConfig.max}
                            step={temperatureConfig.step}
                            value={[localTemperature]}
                            onValueChange={handleTemperatureChange}  // Smooth UI updates
                            onValueCommit={handleTemperatureCommit}  // Backend update on finish
                            className="w-full"
                        />
                    </div>
                    <div className="text-xs text-muted-foreground italic">
                        Higher values make output more creative but less predictable
                    </div>
                </div>
            )}

            {selectedModel.parameters?.reasoning_effort && reasoningEffortOptions && (
                <div className="space-y-2">
                    <Label htmlFor="reasoning-effort">Reasoning Effort</Label>
                    <Select
                        value={reasoningEffort}
                        onValueChange={handleReasoningEffortChange}
                    >
                        <SelectTrigger
                            id="reasoning-effort"
                            className="w-full rounded-xl border-gray-200 bg-white/50 backdrop-blur-sm"
                        >
                            <SelectValue placeholder="Select reasoning effort"/>
                        </SelectTrigger>
                        <SelectContent className="bg-white/95 backdrop-blur-sm border shadow-lg rounded-xl">
                            {reasoningEffortOptions.map(option => (
                                <SelectItem
                                    key={option}
                                    value={option}
                                    className="hover:bg-blue-50/80 focus:bg-blue-50 transition-colors rounded-lg mx-1 my-0.5"
                                >
                                    {option.charAt(0).toUpperCase() + option.slice(1)}
                                </SelectItem>
                            ))}
                        </SelectContent>
                    </Select>
                </div>
            )}
        </div>
    );
};

export default ModelParameterControls;
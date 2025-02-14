import React from 'react';
import {
    Collapsible,
    CollapsibleContent,
    CollapsibleTrigger,
} from "@/components/ui/collapsible";
import {Button} from "@/components/ui/button";
import {ChevronDown, ChevronUp} from 'lucide-react';
import {Card} from '@/components/ui/card';
import PersonaSelector from './PersonaSelector';
import ToolSelector from './ToolSelector';

/**
 * A collapsible options panel that provides persona and tool selection functionality.
 *
 * @component
 * @param {Object} props
 * @param {boolean} props.isOpen - Current open/closed state of the panel
 * @param {Function} props.setIsOpen - Callback to update open/closed state
 * @param {string} props.persona - Currently selected persona name
 * @param {string} props.customPrompt - Custom prompt text
 * @param {number} props.temperature - Model temperature setting
 * @param {string} props.modelName - Current model name
 * @param {Object} props.modelConfigs - Available model configurations
 * @param {string} props.sessionId - Current session identifier
 * @param {Array} props.personas - Available personas
 * @param {Array} props.availableTools - List of available tools
 * @param {Function} props.onEquipTools - Callback for tool equipment changes
 * @param {Array} props.activeTools - Currently equipped tools
 * @param {boolean} props.isReady - Component ready state
 * @param {Object} props.modelParameters - Current model parameters
 * @param {string} props.selectedModel - Selected model identifier
 * @param {Function} props.onUpdateSettings - Settings update callback
 * @param {boolean} props.isInitialized - Component initialization state
 *
 * @returns {React.ReactElement} A collapsible options panel component
 */
const CollapsibleOptions = ({
                                isOpen,
                                setIsOpen,
                                persona,
                                customPrompt,
                                temperature,
                                modelName,
                                modelConfigs,
                                sessionId,
                                personas,
                                availableTools,
                                onEquipTools,
                                activeTools,
                                isReady,
                                modelParameters,
                                selectedModel,
                                onUpdateSettings,
                                isInitialized
                            }) => {
    return (
        <Card className="w-full mb-4 bg-white/50 backdrop-blur-sm border shadow-lg">
            <Collapsible open={isOpen} onOpenChange={setIsOpen}>
                <div className="p-4 flex items-center justify-between">
                    <h2 className="text-lg font-semibold">Options Panel</h2>
                    <CollapsibleTrigger asChild>
                        <Button variant="ghost" size="sm" className="w-9 p-0">
                            {isOpen ? <ChevronUp className="h-4 w-4"/> : <ChevronDown className="h-4 w-4"/>}
                        </Button>
                    </CollapsibleTrigger>
                </div>
                <CollapsibleContent className="px-4 pb-4">
                    <div className="space-y-4">
                        <PersonaSelector
                            persona_name={persona}
                            personas={personas}
                            customPrompt={customPrompt}
                            temperature={temperature}
                            modelName={modelName}
                            modelConfigs={modelConfigs}
                            sessionId={sessionId}
                            modelParameters={modelParameters}
                            selectedModel={selectedModel}
                            onUpdateSettings={onUpdateSettings}
                            isInitialized={isInitialized}
                        />
                        <ToolSelector
                            availableTools={availableTools}
                            onEquipTools={onEquipTools}
                            activeTools={activeTools}
                            sessionId={sessionId}
                            isReady={isReady}
                        />
                    </div>
                </CollapsibleContent>
            </Collapsible>
        </Card>
    );
};

export default CollapsibleOptions;
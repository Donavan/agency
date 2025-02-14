import React, {useState} from 'react';
import {Card, CardContent, CardHeader, CardTitle} from '@/components/ui/card';
import {Checkbox} from '@/components/ui/checkbox';
import {Button} from '@/components/ui/button';
import {ScrollArea} from '@/components/ui/scroll-area';
import {Toast} from '@/components/ui/toast';
import {
    Tooltip,
    TooltipContent,
    TooltipProvider,
    TooltipTrigger,
} from '@/components/ui/tooltip';
import {Badge} from '@/components/ui/badge';
import {Check, AlertCircle} from 'lucide-react';

/**
 * Displays essential tools that cannot be toggled
 * @component
 * @param {Object} props
 * @param {Tool[]} [props.tools=[]] - Array of essential tools
 */
const EssentialTools = ({tools = []}) => (
    <div className="mb-6">
        <h3 className="font-medium mb-3 text-lg text-blue-600">Essential Tools</h3>
        <div className="p-4 bg-blue-50 rounded-lg border border-blue-200">
            <div className="flex flex-wrap gap-2">
                {tools.map((tool) => (
                    <TooltipProvider key={tool.name}>
                        <Tooltip>
                            <TooltipTrigger asChild>
                                <Badge
                                    variant="secondary"
                                    className="px-3 py-1.5 bg-white text-blue-700 border border-blue-200 shadow-sm hover:bg-blue-50 transition-colors"
                                >
                                    <div className="flex items-center gap-1">
                                        <div className="w-2 h-2 rounded-full bg-blue-500"></div>
                                        {tool.name}
                                    </div>
                                </Badge>
                            </TooltipTrigger>
                            <TooltipContent className="bg-white/95 backdrop-blur-sm border shadow-lg rounded-lg">
                                <p className="max-w-xs">{tool.doc || 'No description available'}</p>
                            </TooltipContent>
                        </Tooltip>
                    </TooltipProvider>
                ))}
            </div>
        </div>
    </div>
);

/**
 * Displays a category of tools with selection capabilities
 * @component
 * @param {Object} props
 * @param {string} props.title - Category title
 * @param {Tool[]} props.tools - Tools in this category
 * @param {Set<string>} props.selectedTools - Set of selected tool names
 * @param {string[]} props.activeTools - Array of active tool names
 * @param {(toolName: string) => void} props.onToolToggle - Tool toggle callback
 */
const ToolCategory = ({title, tools = [], selectedTools, activeTools, onToolToggle}) => (
    <div className="mb-6">
        <h3 className="font-medium mb-3 text-lg text-blue-600">{title}</h3>
        <div className="p-4 bg-gray-50 rounded-lg border border-gray-200">
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                {tools.map((tool) => {
                    const isSelected = selectedTools.has(tool.name);
                    const isActive = activeTools.includes(tool.name);
                    return (
                        <TooltipProvider key={tool.name}>
                            <Tooltip>
                                <TooltipTrigger asChild>
                                    <div
                                        className={`flex items-center space-x-2 p-2 rounded-lg transition-colors duration-200 
                      ${isActive ? 'bg-blue-50/80 hover:bg-blue-100/80' : 'hover:bg-white'}`}
                                    >
                                        <Checkbox
                                            checked={isSelected}
                                            onCheckedChange={() => onToolToggle(tool.name)}
                                            id={tool.name}
                                            className="border-gray-300 data-[state=checked]:bg-blue-500 data-[state=checked]:border-blue-500"
                                        />
                                        <label
                                            htmlFor={tool.name}
                                            className={`flex items-center gap-2 text-sm font-medium leading-none cursor-pointer
                        ${isActive ? 'text-blue-700' : 'text-gray-700 hover:text-gray-900'}`}
                                        >
                                            {tool.name}
                                            {isActive && (
                                                <span
                                                    className="inline-flex items-center text-xs text-green-600 bg-green-50 px-2 py-0.5 rounded-full border border-green-200">
                          <Check className="w-3 h-3 mr-1"/>
                          Active
                        </span>
                                            )}
                                        </label>
                                    </div>
                                </TooltipTrigger>
                                <TooltipContent className="bg-white/95 backdrop-blur-sm border shadow-lg rounded-lg">
                                    <p className="max-w-xs">{tool.doc || 'No description available'}</p>
                                </TooltipContent>
                            </Tooltip>
                        </TooltipProvider>
                    );
                })}
            </div>
        </div>
    </div>
);

/**
 * ToolSelector is a component that manages the selection and equipment of tools for an agent.
 *
 * @component
 * @param {Object} props
 * @param {Object} props.availableTools - Configuration object containing tool categories and groups
 * @param {Array} props.availableTools.essential_tools - Array of essential tool objects
 * @param {Array} props.availableTools.categories - Array of category names
 * @param {Object} props.availableTools.groups - Object mapping categories to tool arrays
 * @param {Function} props.onEquipTools - Callback function when tools are equipped
 * @param {Array} [props.activeTools=[]] - Array of currently active tool names
 * @param {string} props.sessionId - Current session identifier
 * @param {boolean} props.isReady - Flag indicating if the agent is ready
 */
const ToolSelector = ({availableTools, onEquipTools, activeTools = [], sessionId, isReady}) => {
    // Local UI state
    const [selectedTools, setSelectedTools] = useState(new Set(activeTools));
    const [error, setError] = useState(null);
    const [toast, setToast] = useState(null);
    const [isLoading, setIsLoading] = useState(false);

    /**
     * Displays a toast notification
     * @param {string} message - Message to display
     * @param {'success' | 'error'} [type='success'] - Type of toast
     */
    const showToast = (message, type = 'success') => {
        setToast({message, type});
        setTimeout(() => setToast(null), 3000);
    };


    /**
     * Toggles a tool's selection state
     * @param {string} toolName - Name of the tool to toggle
     */
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

    /**
     * Handles the equipping of selected tools
     * @async
     * @throws {Error} When tool equipment fails
     */
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

    // Render waiting message if not ready
    if (!sessionId || !isReady) {
        return (
            <Card className="w-full">
                <CardHeader>
                    <CardTitle>Available Tools</CardTitle>
                </CardHeader>
                <CardContent>
                    <div className="flex items-center gap-2 justify-center p-4 text-amber-600">
                        <AlertCircle className="h-5 w-5"/>
                        <p>Waiting for agent initialization...</p>
                    </div>
                </CardContent>
            </Card>
        );
    }

    // Render no tools message if none available
    if (!availableTools || !Object.keys(availableTools).length) {
        return (
            <Card className="w-full">
                <CardHeader>
                    <CardTitle>Available Tools</CardTitle>
                </CardHeader>
                <CardContent>
                    <div className="flex items-center justify-center p-4">
                        <p className="text-muted-foreground">No tools available</p>
                    </div>
                </CardContent>
            </Card>
        );
    }

    return (
        <>
            <Card className="w-full">
                <CardHeader>
                    <CardTitle className="flex justify-between items-center">
                        <span>Available Tools</span>
                        {isLoading && (
                            <span className="text-sm text-muted-foreground">Updating...</span>
                        )}
                    </CardTitle>
                </CardHeader>
                <CardContent>
                    {error && (
                        <div className="mb-4 p-3 bg-red-50 text-red-600 rounded-lg border border-red-200">
                            {error}
                        </div>
                    )}
                    <ScrollArea className="h-[400px] pr-4">
                        <div className="space-y-6">
                            <EssentialTools tools={availableTools.essential_tools}/>
                            {availableTools.categories.map((category) => (
                                <ToolCategory
                                    key={category}
                                    title={category}
                                    tools={availableTools.groups[category] || []}
                                    selectedTools={selectedTools}
                                    activeTools={activeTools}
                                    onToolToggle={handleToolToggle}
                                />
                            ))}
                        </div>
                    </ScrollArea>
                    <Button
                        onClick={handleEquipTools}
                        disabled={!isReady || isLoading}
                        className="w-full mt-4 bg-blue-500 hover:bg-blue-600 text-white rounded-full font-medium shadow-sm transition-all hover:shadow-md disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                        {isLoading ? 'Updating Tools...' : 'Equip Selected Tools'}
                    </Button>
                </CardContent>
            </Card>
            {toast && (
                <Toast
                    message={toast.message}
                    type={toast.type}
                    onClose={() => setToast(null)}
                />
            )}
        </>
    );
};

export default ToolSelector;
import React from 'react';
import { Button } from '@/components/ui/button';
import AgentConfigDisplay from './AgentConfigDisplay';
import { Trash2, Activity, Wrench, Info } from 'lucide-react';
import {
    AlertDialog,
    AlertDialogAction,
    AlertDialogCancel,
    AlertDialogContent,
    AlertDialogDescription,
    AlertDialogFooter,
    AlertDialogHeader,
    AlertDialogTitle,
    AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import {
    Tooltip,
    TooltipContent,
    TooltipProvider,
    TooltipTrigger,
} from "@/components/ui/tooltip";

const API_URL = import.meta.env.VITE_API_URL;

const StatusBar = ({
    isReady,
    activeTools = [],
    onSessionsDeleted,
    isInitializing = false,
    isProcessing = false,
    sessionId,
    settingsVersion
}) => {
    const handleDeleteSessions = async () => {
        try {
            const response = await fetch(`${API_URL}/sessions`, {
                method: 'DELETE',
            });

            if (!response.ok) {
                throw new Error('Failed to delete sessions');
            }

            const data = await response.json();
            if (onSessionsDeleted) {
                onSessionsDeleted();
            }
        } catch (error) {
            console.error('Error deleting sessions:', error);
        }
    };

    const getStatusInfo = () => {
        // console.log('StatusBar props:', {
        //     isReady,
        //     isInitializing,
        //     isProcessing,
        //     sessionId,
        //     settingsVersion
        // });
        if (isProcessing) {
            return {
                message: 'Processing...',
                color: 'text-red-500',
                description: 'Agent is processing your request',
                iconClass: 'animate-pulse'
            };
        }
        if (isInitializing) {
            return {
                message: 'Initializing Application...',
                color: 'text-yellow-500',
                description: 'Loading initial application data',
                iconClass: ''
            };
        }
        if (!isReady) {
            return {
                message: 'Initializing Agent...',
                color: 'text-yellow-500',
                description: 'Setting up the agent and tools',
                iconClass: ''
            };
        }
        return {
            message: 'Ready',
            color: 'text-green-500',
            description: 'System is ready to process requests',
            iconClass: ''
        };
    };

    const statusInfo = getStatusInfo();

    return (
        <div className="flex items-center justify-between p-4 bg-white/95 backdrop-blur-sm rounded-lg shadow-sm border">
            <div className="flex items-center space-x-6">
                <div className="flex items-center space-x-2">
                    <TooltipProvider>
                        <Tooltip>
                            <TooltipTrigger asChild>
                                <div className="flex items-center space-x-2">
                                    <Activity
                                        className={`w-4 h-4 ${statusInfo.color} ${statusInfo.iconClass} ${
                                            isProcessing ? 'scale-110 transition-transform duration-200' : ''
                                        }`}
                                    />
                                    <span className="text-sm font-medium">
                                        Status: {statusInfo.message}
                                    </span>
                                </div>
                            </TooltipTrigger>
                            <TooltipContent>
                                <p>{statusInfo.description}</p>
                            </TooltipContent>
                        </Tooltip>
                    </TooltipProvider>
                    {isReady && sessionId && (
                        <AgentConfigDisplay
                            sessionId={sessionId}
                            className="ml-2"
                            settingsVersion={settingsVersion}
                        />
                    )}
                </div>

                {isReady && activeTools && activeTools.length > 0 && (
                    <div className="flex items-center space-x-2 px-4 py-1.5 bg-blue-50 rounded-full border border-blue-100">
                        <Wrench className="w-4 h-4 text-blue-500" />
                        <span className="text-sm text-blue-700">
                            Active Tools: {activeTools.length}
                        </span>
                        <TooltipProvider>
                            <Tooltip>
                                <TooltipTrigger asChild>
                                    <Info className="w-4 h-4 text-blue-400 cursor-help" />
                                </TooltipTrigger>
                                <TooltipContent className="bg-white border shadow-md">
                                    <p className="max-w-xs text-gray-700">{activeTools.join(', ')}</p>
                                </TooltipContent>
                            </Tooltip>
                        </TooltipProvider>
                    </div>
                )}
            </div>

            {isReady && (
                <AlertDialog>
                    <AlertDialogTrigger asChild>
                        <Button
                            variant="destructive"
                            size="sm"
                            className="flex items-center space-x-2 bg-red-500 hover:bg-red-600 rounded-full"
                        >
                            <Trash2 className="w-4 h-4" />
                            <span>Delete All Sessions</span>
                        </Button>
                    </AlertDialogTrigger>
                    <AlertDialogContent className="bg-white">
                        <AlertDialogHeader>
                            <AlertDialogTitle>Are you sure?</AlertDialogTitle>
                            <AlertDialogDescription>
                                This action will delete all active chat sessions and cannot be undone.
                            </AlertDialogDescription>
                        </AlertDialogHeader>
                        <AlertDialogFooter>
                            <AlertDialogCancel className="border-gray-200 rounded-full">Cancel</AlertDialogCancel>
                            <AlertDialogAction
                                onClick={handleDeleteSessions}
                                className="bg-red-500 hover:bg-red-600 text-white rounded-full"
                            >
                                Delete
                            </AlertDialogAction>
                        </AlertDialogFooter>
                    </AlertDialogContent>
                </AlertDialog>
            )}
        </div>
    );
};

export default StatusBar;
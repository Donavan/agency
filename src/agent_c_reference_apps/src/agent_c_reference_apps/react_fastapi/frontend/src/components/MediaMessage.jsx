import React, {useState} from 'react';
import {Monitor, ChevronDown, Maximize2, Info} from 'lucide-react';
import {Dialog, DialogContent} from "@/components/ui/dialog";
import {Card} from "@/components/ui/card";

const MediaMessage = ({message}) => {
    const [isExpanded, setIsExpanded] = useState(true);
    const [isFullscreen, setIsFullscreen] = useState(false);

    if (!message?.content || !message?.contentType) {
        console.warn('MediaMessage: Invalid message structure', message);
        return null;
    }

    const toggleFullscreen = (e) => {
        e.stopPropagation();
        setIsFullscreen(!isFullscreen);
    };

    const MediaContentWrapper = ({children, allowFullscreen = false}) => (
        <div className="relative">
            {allowFullscreen && (
                <button
                    onClick={toggleFullscreen}
                    className="absolute top-2 right-2 p-1 rounded-lg bg-white/80 hover:bg-white shadow-sm transition-colors"
                >
                    <Maximize2 className="h-4 w-4 text-blue-600"/>
                </button>
            )}
            {children}
        </div>
    );

    const renderContent = () => {
        if (message.contentType === 'image/svg+xml') {
            return (
                <MediaContentWrapper allowFullscreen>
                    <div
                        className="w-full"
                        dangerouslySetInnerHTML={{__html: message.content}}
                    />
                </MediaContentWrapper>
            );
        }

        if (message.contentType === "text/html") {
            return (
                <MediaContentWrapper>
                    <div dangerouslySetInnerHTML={{__html: message.content}}/>
                </MediaContentWrapper>
            );
        }

        if (message.contentType?.startsWith('image/')) {
            return (
                <MediaContentWrapper allowFullscreen>
                    <img
                        src={`data:${message.contentType};base64,${message.content}`}
                        alt="Generated Content"
                        className="w-full h-auto rounded shadow-lg object-contain"
                        loading="lazy"
                    />
                </MediaContentWrapper>
            );
        }

        return null;
    };

    const renderMetadata = () => {
        if (!message.metadata) return null;

        const {sent_by_class, sent_by_function} = message.metadata;
        if (!sent_by_class && !sent_by_function) return null;

        return (
            <div className="mt-4 pt-3 border-t border-amber-100 flex items-center gap-2 text-xs text-amber-600">
                <Info className="h-4 w-4 shrink-0" />
                <span className="font-medium">
                    {sent_by_class}
                    {sent_by_class && sent_by_function && <span className="mx-2">•</span>}
                    {sent_by_function}
                </span>
            </div>
        );
    };

    const FullscreenDialog = () => (
        <Dialog open={isFullscreen} onOpenChange={setIsFullscreen}>
            <DialogContent className="p-0 overflow-hidden max-w-7xl">
                <div className="w-full h-full flex items-center justify-center bg-white p-4">
                    {message.contentType === 'image/svg+xml' ? (
                        <div
                            className="w-full h-full flex items-center justify-center"
                            dangerouslySetInnerHTML={{__html: message.content}}
                        />
                    ) : message.contentType?.startsWith('image/') ? (
                        <img
                            src={`data:${message.contentType};base64,${message.content}`}
                            alt="Generated content"
                            className="max-w-full max-h-[80vh] object-contain rounded shadow-lg"
                        />
                    ) : (
                        <div dangerouslySetInnerHTML={{__html: message.content}}/>
                    )}
                </div>
            </DialogContent>
        </Dialog>
    );

    const getContentTypeDisplay = () => {
        return message.contentType.split('/')[1].toUpperCase();
    };

    return (
        <>
            <Card className="bg-white shadow-sm overflow-hidden">
                <div
                    className="flex items-center justify-between px-4 py-3 cursor-pointer bg-amber-50 hover:bg-amber-100 transition-colors"
                    onClick={() => setIsExpanded(!isExpanded)}
                >
                    <div className="flex items-center gap-2">
                        <Monitor className="h-5 w-5 text-amber-600"/>
                        <span className="font-semibold text-amber-800">
                            {getContentTypeDisplay()}
                        </span>
                    </div>
                    <ChevronDown
                        className={`h-5 w-5 text-amber-600 transform transition-transform duration-200 ${
                            isExpanded ? "rotate-180" : ""
                        }`}
                    />
                </div>

                {isExpanded && (
                    <div className="border-t border-amber-100 p-4">
                        <div className="max-h-96 overflow-auto">
                            {renderContent()}
                        </div>
                        {renderMetadata()}
                    </div>
                )}
            </Card>

            {isFullscreen && <FullscreenDialog/>}
        </>
    );
};

export default MediaMessage;
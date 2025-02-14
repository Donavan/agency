import React from 'react';
import { Cpu } from 'lucide-react';

const TokenUsageDisplay = ({ usage }) => {
    if (!usage || (!usage.prompt_tokens && !usage.completion_tokens)) return null;

    const {
        prompt_tokens = 0,
        completion_tokens = 0,
        total_tokens = prompt_tokens + completion_tokens
    } = usage;

    return (
        <div className="mt-2 flex items-center gap-2 text-xs text-gray-400 border-t border-gray-100 pt-2">
            <Cpu className="h-3 w-3" />
            <div className="flex gap-3">
                <span title="Tokens used in the prompt">Prompt: {prompt_tokens.toLocaleString()}</span>
                <span>·</span>
                <span title="Tokens used in the completion">Completion: {completion_tokens.toLocaleString()}</span>
                <span>·</span>
                <span title="Total tokens used">Total: {total_tokens.toLocaleString()}</span>
            </div>
        </div>
    );
};

export default TokenUsageDisplay;
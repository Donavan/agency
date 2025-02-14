import React, { useState } from "react";
import ToolCallItem from "./ToolCallItem";
import { Wrench } from "lucide-react";
import { Badge } from "@/components/ui/badge";

const ToolCallDisplay = ({ toolCalls }) => {
  const [isExpanded, setIsExpanded] = useState(false); // Default to collapsed

  // Ensure toolCalls is an array and filter out any null values
  const validTools = Array.isArray(toolCalls) ? toolCalls.filter(tool => tool) : [];
  const toolCount = validTools.length;

  if (toolCount === 0) return null;

  return (
    <div className="border rounded-2xl my-2 bg-white shadow-lg overflow-hidden">
      <div
        className="flex items-center justify-between p-4 cursor-pointer bg-blue-100 hover:bg-blue-200 transition-colors rounded-t-2xl"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center gap-3">
          <Wrench className="h-5 w-5 text-blue-500" />
          <h4 className="text-blue-700 font-medium">Tool Calls</h4>
          <Badge variant="secondary" className="bg-blue-200 hover:bg-blue-200 text-blue-700">
            {toolCount}
          </Badge>
        </div>
        <span className="text-blue-500">{isExpanded ? "▲" : "▼"}</span>
      </div>

      {isExpanded && (
        <div className="p-4 space-y-2 rounded-b-2xl bg-white">
          {validTools.map((toolCall, idx) => (
            <ToolCallItem
              key={toolCall.id || idx}
              tool={{
                name: toolCall.name || toolCall.function?.name,
                arguments: toolCall.arguments || toolCall.function?.arguments,
                id: toolCall.id || toolCall.tool_call_id
              }}
              results={toolCall.results}
            />
          ))}
        </div>
      )}
    </div>
  );
};

export default ToolCallDisplay;
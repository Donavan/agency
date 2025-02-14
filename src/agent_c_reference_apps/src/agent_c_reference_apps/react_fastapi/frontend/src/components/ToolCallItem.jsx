import React, { useState } from 'react';
import { PocketKnife, ChevronDown } from 'lucide-react';

function formatData(data) {
  if (!data) return "";

  try {
    if (typeof data === "string") {
      const parsed = JSON.parse(data);
      return JSON.stringify(parsed, null, 2);
    }
    return JSON.stringify(data, null, 2);
  } catch (e) {
    return String(data);
  }
}

const ToolCallItem = ({ tool, results }) => {
  const [isExpanded, setIsExpanded] = useState(false); // Default to collapsed

  if (!tool?.name) return null;

  const formattedArguments = tool.arguments ? formatData(tool.arguments) : "";
  const formattedResults = results ? formatData(results) : "";

  return (
    <div className="border border-blue-200 rounded-lg mb-2 bg-white shadow-sm overflow-hidden">
      <div
        className="flex items-center justify-between px-4 py-3 cursor-pointer bg-blue-50 hover:bg-blue-100 transition-colors rounded-lg"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center gap-2">
          <PocketKnife className="h-5 w-5 text-blue-600" />
          <span className="font-semibold text-blue-800">
            {tool.name}
          </span>
        </div>
        <ChevronDown
          className={`h-5 w-5 text-blue-600 transform transition-transform duration-200 ${
            isExpanded ? "rotate-180" : ""
          }`}
        />
      </div>

      {isExpanded && (
        <div className="border-t border-blue-200 bg-white p-4 rounded-b-lg">
          {formattedArguments && (
            <div className="mb-4">
              <h5 className="text-sm font-semibold text-blue-600 mb-2">Arguments:</h5>
              <pre className="text-sm font-mono bg-blue-50 p-2 rounded-md whitespace-pre-wrap overflow-x-auto">
                {formattedArguments}
              </pre>
            </div>
          )}

          {formattedResults && (
            <div>
              <h5 className="text-sm font-semibold text-blue-600 mb-2">Results:</h5>
              <pre className="text-sm font-mono bg-blue-50 p-2 rounded-md whitespace-pre-wrap overflow-x-auto">
                {formattedResults}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default ToolCallItem;
import React, { useMemo } from 'react';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';

const MarkdownMessage = ({ content }) => {
  // Memoize the markdown content to prevent unnecessary re-renders
  const memoizedContent = useMemo(() => {
    // Ensure proper spacing around headers
    let processedContent = content.replace(/\n(#{1,6})\s*([^\n]+)/g, '\n\n$1 $2\n');

    // Ensure proper list formatting
    processedContent = processedContent.replace(/^\s*[-*]\s/gm, '* ');

    // Add extra newline before lists
    processedContent = processedContent.replace(/\n[-*]/g, '\n\n*');

    return processedContent || '';
  }, [content]);

  return (
    <div className="prose prose-sm max-w-none prose-headings:mt-4 prose-headings:mb-2">
      <ReactMarkdown
        components={{
          h1: ({node, ...props}) => <h1 className="text-2xl font-bold mt-6 mb-4" {...props} />,
          h2: ({node, ...props}) => <h2 className="text-xl font-bold mt-5 mb-3" {...props} />,
          h3: ({node, ...props}) => <h3 className="text-lg font-bold mt-4 mb-2" {...props} />,
          h4: ({node, ...props}) => <h4 className="text-base font-bold mt-3 mb-2" {...props} />,
          // Enhanced list rendering
          ul: ({node, ...props}) => (
            <ul className="list-disc ml-4 mt-2 mb-4 space-y-2" {...props} />
          ),
          li: ({node, ...props}) => (
            <li className="mt-1" {...props} />
          ),
          // Enhanced bold text rendering
          strong: ({node, ...props}) => (
            <strong className="font-bold text-purple-800" {...props} />
          ),
          // Enhanced code block rendering
          code: ({node, inline, className, children, ...props}) => {
            const match = /language-(\w+)/.exec(className || '');
            const language = match ? match[1] : '';

            return !inline ? (
              <div className="rounded-lg overflow-hidden my-4">
                <SyntaxHighlighter
                  style={oneDark}
                  language={language || 'text'}
                  PreTag="div"
                  className="!my-0"
                  showLineNumbers={true}
                  {...props}
                >
                  {String(children).replace(/\n$/, '')}
                </SyntaxHighlighter>
              </div>
            ) : (
              <code className="bg-purple-100 px-1 rounded text-purple-800" {...props}>
                {children}
              </code>
            );
          },
          // Enhanced blockquote rendering
          blockquote: ({node, ...props}) => (
            <blockquote className="border-l-4 border-purple-300 pl-4 my-4 italic" {...props} />
          ),
          // Enhance paragraph rendering
          p: ({node, ...props}) => (
            <p className="my-2 leading-relaxed" {...props} />
          ),
          // Enhanced horizontal rule
          hr: ({node, ...props}) => (
            <hr className="my-6 border-t-2 border-purple-100" {...props} />
          ),
        }}
      >
        {memoizedContent}
      </ReactMarkdown>
    </div>
  );
};

export default MarkdownMessage;
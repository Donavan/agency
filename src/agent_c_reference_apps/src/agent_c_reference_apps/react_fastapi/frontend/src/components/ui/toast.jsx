import * as React from "react"
import { X } from "lucide-react"
import { cn } from "@/lib/utils"

const Toast = React.forwardRef(({ className, type = "success", message, onClose, ...props }, ref) => {
  const baseClasses = "fixed bottom-4 right-4 p-4 rounded-lg shadow-lg flex items-center justify-between gap-2 min-w-[300px] transition-all transform translate-y-0 duration-300 ease-out z-50";

  const variants = {
    success: "bg-green-100 text-green-800 border border-green-200",
    error: "bg-red-100 text-red-800 border border-red-200",
    info: "bg-blue-100 text-blue-800 border border-blue-200"
  };

  return (
    <div
      ref={ref}
      className={cn(baseClasses, variants[type], className)}
      {...props}
    >
      <span className="flex-1 text-sm font-medium">{message}</span>
      <button
        onClick={onClose}
        className="text-gray-500 hover:text-gray-700 transition-colors"
      >
        <X className="h-4 w-4" />
      </button>
    </div>
  )
})
Toast.displayName = "Toast"

export { Toast }
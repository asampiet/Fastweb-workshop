"use client";
import { useState, useRef, useEffect, FormEvent } from "react";
import MessageBubble from "./MessageBubble";
import { useMetrics } from "@/context/MetricsContext";

interface Message {
  role: "user" | "agent";
  content: string;
}

export default function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const { setMetrics } = useMetrics();
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    const text = input.trim();
    if (!text || isLoading) return;

    setInput("");
    setMessages((prev) => [...prev, { role: "user", content: text }]);
    setIsLoading(true);

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text }),
      });
      const data = await res.json();
      setMessages((prev) => [...prev, { role: "agent", content: data.response || data.error || "No response" }]);
      if (data.metrics) setMetrics(data.metrics);
    } catch {
      setMessages((prev) => [...prev, { role: "agent", content: "Error: Could not reach the backend." }]);
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div className="flex flex-col h-full">
      <div className="bg-gray-800 text-white px-4 py-3 text-sm font-semibold">
        🔧 Telco RCA Agent — Smart MCP Workshop
      </div>
      <div className="flex-1 overflow-y-auto p-4">
        {messages.length === 0 && (
          <p className="text-gray-400 text-center mt-10">
            Describe a network issue to start diagnosis...
          </p>
        )}
        {messages.map((m, i) => (
          <MessageBubble key={i} role={m.role} content={m.content} />
        ))}
        {isLoading && (
          <div className="flex justify-start mb-3">
            <div className="bg-gray-100 rounded-lg px-4 py-3 text-sm text-gray-500 animate-pulse">
              Analyzing network telemetry...
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>
      <form onSubmit={handleSubmit} className="border-t p-3 flex gap-2">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="e.g. Subscriber 999123 complains YouTube buffers constantly..."
          className="flex-1 border rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          disabled={isLoading}
        />
        <button
          type="submit"
          disabled={isLoading || !input.trim()}
          className="bg-blue-600 text-white px-4 py-2 rounded-lg text-sm font-medium disabled:opacity-50"
        >
          Send
        </button>
      </form>
    </div>
  );
}

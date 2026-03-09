"use client";
import ChatInterface from "@/components/ChatInterface";
import ComparisonPanel from "@/components/ComparisonPanel";

export default function Home() {
  return (
    <div className="h-screen flex flex-col lg:flex-row">
      <div className="flex-1 lg:w-2/3 flex flex-col min-h-0">
        <ChatInterface />
      </div>
      <div className="lg:w-1/3 border-t lg:border-t-0 lg:border-l overflow-y-auto">
        <ComparisonPanel />
      </div>
    </div>
  );
}

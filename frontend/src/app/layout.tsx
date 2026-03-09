import "./globals.css";
import { MetricsProvider } from "@/context/MetricsContext";

export const metadata = { title: "Telco RCA Agent", description: "Smart MCP Workshop" };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <MetricsProvider>{children}</MetricsProvider>
      </body>
    </html>
  );
}

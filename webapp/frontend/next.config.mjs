/** @type {import('next').NextConfig} */
const nextConfig = {
  // Proxy API calls to the FastAPI backend so the browser never sees a cross-origin request.
  async rewrites() {
    return [{ source: "/api/:path*", destination: "http://127.0.0.1:8000/api/:path*" }];
  },
};
export default nextConfig;

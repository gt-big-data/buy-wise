// Set at build time by webpack. To point at a deployed backend:
//   BACKEND_URL=https://api.example.com npm run build
declare const __BACKEND_URL__: string;

export const BACKEND_URL: string = __BACKEND_URL__;
